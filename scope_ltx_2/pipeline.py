"""LTX 2.3 audio-video generation pipeline for Daydream Scope.

Uses Kijai's separated ComfyUI-format checkpoints (distilled v3 FP8) with
a standalone inference implementation adapted from ComfyUI's lightricks code.

Memory Management (24GB GPU):
    - Gemma 3 12B FP8 text encoder: ~13GB on GPU, offloaded after encoding
    - Text projection (aggregate embeds): ~1.5GB, runs on GPU briefly
    - Transformer (22B FP8): ~23GB total, CPU-resident. Scaffold (norms,
      embeddings, etc.) persists on GPU; transformer blocks streamed from
      CPU with double-buffered async transfers and prefetching.
    - VAEs: kept resident on GPU (~1GB total). The streaming config
      automatically accounts for their memory at setup time.
    - Between generations: streaming state (hooks, pinned memory) persists.
      Resident blocks are temporarily offloaded for VAE decode, then
      reloaded by pre-forward hooks on the next denoising pass.
    - Full teardown only when text encoder needs to reload (prompt change).

Audio Support:
    LTX 2.3 natively supports synchronized audio-video generation.
"""

import gc
import json
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_model_file_path
from scope.core.pipelines.interface import Pipeline

from .schema import LTX2Config

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig

logging.getLogger("scope_ltx_2").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def _load_sd_with_prefix_replace(
    checkpoint_path: str,
    prefix_map: dict[str, str],
    dtype: torch.dtype | None = None,
) -> dict:
    """Load a safetensors file, keeping only keys matching given prefixes and
    replacing those prefixes. Mirrors ComfyUI's state_dict_prefix_replace.

    Args:
        checkpoint_path: Path to .safetensors file.
        prefix_map: {old_prefix: new_prefix} — a key is included if it starts
            with any old_prefix; the longest matching prefix is replaced.
        dtype: Optional dtype cast for values.
    """
    from safetensors import safe_open

    sd: dict[str, torch.Tensor] = {}
    sorted_prefixes = sorted(prefix_map.keys(), key=len, reverse=True)
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            for prefix in sorted_prefixes:
                if key.startswith(prefix):
                    new_key = prefix_map[prefix] + key[len(prefix):]
                    v = f.get_tensor(key)
                    sd[new_key] = v.to(dtype=dtype) if dtype is not None else v
                    break
    return sd

DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]


def _log_gpu_memory(stage: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU [{stage}]: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def _move_module_to(module: torch.nn.Module, device, non_blocking=False):
    """Move all parameters and buffers to device, preserving dtypes."""
    for param in module.parameters():
        param.data = param.data.to(device=device, non_blocking=non_blocking)
    for buf in module.buffers():
        buf.data = buf.data.to(device=device, non_blocking=non_blocking)


class LTX2Pipeline(Pipeline):
    """LTX 2.3 audio-video generation pipeline."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LTX2Config

    def __init__(
        self,
        height: int = 512,
        width: int = 768,
        num_frames: int = 33,
        frame_rate: float = 24.0,
        randomize_seed: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        transformer_path: str | None = None,
        text_projection_path: str | None = None,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        gemma_model_path: str | None = None,
        gemma_tokenizer_path: str | None = None,
        ffn_chunk_size: int | None = 4096,
        **kwargs,
    ):
        from .model_loader import load_transformer
        from .text_encoder import TextEmbeddingProjection, load_gemma_text_encoder

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.randomize_seed = randomize_seed
        self.ffn_chunk_size = ffn_chunk_size

        if kwargs:
            logger.debug(f"LTX2Pipeline ignoring unknown kwargs: {list(kwargs.keys())}")

        start = time.time()

        # Resolve model paths
        kijai_dir = get_model_file_path("LTX2.3_comfy")
        comfy_dir = get_model_file_path("ltx-2")

        if transformer_path is None:
            transformer_path = str(kijai_dir / "diffusion_models" / "ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors")
        if text_projection_path is None:
            text_projection_path = str(kijai_dir / "text_encoders" / "ltx-2.3_text_projection_bf16.safetensors")
        if video_vae_path is None:
            video_vae_path = str(kijai_dir / "vae" / "LTX23_video_vae_bf16.safetensors")
        if audio_vae_path is None:
            audio_vae_path = str(kijai_dir / "vae" / "LTX23_audio_vae_bf16.safetensors")
        if gemma_model_path is None:
            fp8_path = comfy_dir / "split_files" / "text_encoders" / "gemma_3_12B_it_fp8_scaled.safetensors"
            if fp8_path.exists():
                gemma_model_path = str(fp8_path)
            else:
                logger.warning("FP8 Gemma not found, falling back to bf16 from gemma-3-12b-it")
                gemma_model_path = str(get_model_file_path("gemma-3-12b-it"))
        if gemma_tokenizer_path is None:
            gemma_tokenizer_path = str(get_model_file_path("gemma-3-12b-it"))

        logger.info(f"Transformer: {transformer_path}")
        logger.info(f"Text projection: {text_projection_path}")
        logger.info(f"Gemma: {gemma_model_path}")
        logger.info(f"Gemma tokenizer: {gemma_tokenizer_path}")

        # Step 1: Load Gemma on GPU (FP8 ~13GB, bf16 ~23GB)
        gemma_device = device if Path(gemma_model_path).suffix == ".safetensors" else torch.device("cpu")
        logger.info(f"Loading Gemma 3 12B on {gemma_device}...")
        self._text_encoder, self._tokenizer = load_gemma_text_encoder(
            gemma_model_path, gemma_tokenizer_path, device=gemma_device, dtype=dtype
        )
        self._text_encoder_on_gpu = True
        _log_gpu_memory("gemma loaded")

        # Step 2: Load text projection on CPU
        logger.info("Loading text projection (aggregate embeds)...")
        self._text_projection = TextEmbeddingProjection.from_checkpoint(
            text_projection_path, dtype=dtype
        )

        # Step 3: Load transformer to CPU
        logger.info("Loading transformer (FP8, CPU-resident)...")
        self._transformer = load_transformer(transformer_path, device=torch.device("cpu"), dtype=dtype)

        if ffn_chunk_size is not None:
            self._apply_ffn_chunking(ffn_chunk_size)

        _log_gpu_memory("transformer loaded (CPU)")

        # Step 4: Load VAEs to CPU
        logger.info("Loading video VAE...")
        self._video_vae = self._load_video_vae(video_vae_path, dtype)

        logger.info("Loading audio VAE...")
        self._audio_vae, self._vocoder = self._load_audio_vae(audio_vae_path, dtype)

        # Prompt cache
        self._cached_prompt_text = None
        self._cached_context = None
        self._streaming_state = None
        self._vaes_on_gpu = False

        logger.info(f"LTX 2.3 pipeline loaded in {time.time() - start:.1f}s")
        _log_gpu_memory("all loaded")

    # ------------------------------------------------------------------
    # Model management helpers
    # ------------------------------------------------------------------

    def _apply_ffn_chunking(self, chunk_size: int):
        for block in self._transformer.transformer_blocks:
            if hasattr(block, 'ff'):
                block.ff = _ChunkedFFN(block.ff, chunk_size)
            if hasattr(block, 'audio_ff'):
                block.audio_ff = _ChunkedFFN(block.audio_ff, chunk_size)
        logger.info(f"FFN chunking applied with chunk_size={chunk_size}")

    def _offload_text_encoder(self):
        if not self._text_encoder_on_gpu:
            return
        logger.info("Offloading Gemma to CPU...")
        _move_module_to(self._text_encoder, "cpu")
        self._text_encoder_on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()
        _log_gpu_memory("gemma offloaded")

    def _load_text_encoder(self):
        if self._text_encoder_on_gpu:
            return
        logger.info("Loading Gemma back to GPU...")
        _move_module_to(self._text_encoder, self.device)
        self._text_encoder_on_gpu = True
        _log_gpu_memory("gemma reloaded")

    def _move_vaes_to_gpu(self):
        """Move all VAE models to GPU and keep them resident.

        Called before denoising setup so that calculate_optimal_streaming_config
        sees the VAE memory as already allocated and reserves fewer resident
        transformer blocks accordingly.
        """
        if self._vaes_on_gpu:
            return
        logger.info("Moving VAEs to GPU (will remain resident)...")
        _move_module_to(self._video_vae, self.device)
        _move_module_to(self._audio_vae, self.device)
        _move_module_to(self._vocoder, self.device)
        self._vaes_on_gpu = True
        _log_gpu_memory("VAEs on GPU")

    def _offload_vaes(self):
        """Offload all VAE models to CPU to free VRAM for text encoder."""
        if not self._vaes_on_gpu:
            return
        logger.info("Offloading VAEs to CPU...")
        _move_module_to(self._video_vae, "cpu")
        _move_module_to(self._audio_vae, "cpu")
        _move_module_to(self._vocoder, "cpu")
        self._vaes_on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()
        _log_gpu_memory("VAEs offloaded")

    def _load_video_vae(self, checkpoint_path: str, dtype: torch.dtype):
        """Load video VAE using ComfyUI-compatible VideoVAE architecture.

        Kijai's checkpoint stores the full VAE (encoder + decoder + per_channel_statistics).
        The config is embedded in the safetensors metadata under the "vae" key.
        """
        from safetensors import safe_open
        from .comfy_vae import VideoVAE

        with safe_open(checkpoint_path, framework="pt") as f:
            full_config = json.loads(f.metadata()["config"])

        vae_config = full_config.get("vae", full_config)
        vae = VideoVAE(config=vae_config)

        sd = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                v = f.get_tensor(key)
                sd[key] = v.to(dtype=dtype) if dtype is not None else v
        vae.load_state_dict(sd, strict=False, assign=True)
        vae = vae.to(device="cpu", dtype=dtype).eval()
        logger.info(f"Video VAE loaded: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M params")
        return vae

    def _load_audio_vae(self, checkpoint_path: str, dtype: torch.dtype):
        """Load audio VAE + vocoder following ComfyUI's AudioVAE pattern.

        Matches the loading strategy from ComfyUI's AudioVAE.__init__:
        - No dtype casting during load (preserves float32 precision)
        - No assign=True (copies values into float32-initialized parameters)
        - Signal processing buffers (STFT basis, mel filterbank) stay float32

        Kijai's checkpoint keys:
          audio_vae.encoder.*, audio_vae.decoder.*, audio_vae.per_channel_statistics.*
          vocoder.vocoder.*, vocoder.bwe_generator.*, vocoder.mel_stft.*, vocoder.resampler.*
        """
        from safetensors import safe_open
        from .comfy_vae import CausalAudioAutoencoder, Vocoder, VocoderWithBWE

        with safe_open(checkpoint_path, framework="pt") as f:
            config = json.loads(f.metadata()["config"])

        # Audio VAE — load without dtype/assign to keep float32 precision
        audio_vae_config = config.get("audio_vae", config)
        audio_vae = CausalAudioAutoencoder(config=audio_vae_config)
        sd = _load_sd_with_prefix_replace(checkpoint_path, {"audio_vae.": ""})
        audio_vae.load_state_dict(sd, strict=False)
        audio_vae = audio_vae.to(device="cpu").eval()
        logger.info(f"Audio VAE loaded: {sum(p.numel() for p in audio_vae.parameters())/1e6:.1f}M params")

        # Vocoder — load without dtype/assign (STFT basis and mel filterbank
        # need float32 precision for correct mel spectrogram computation)
        vocoder_config = config.get("vocoder", {})
        if "bwe" in vocoder_config:
            vocoder = VocoderWithBWE(config=vocoder_config)
        else:
            vocoder = Vocoder(config=vocoder_config)
        sd = _load_sd_with_prefix_replace(checkpoint_path, {"vocoder.": ""})
        vocoder.load_state_dict(sd, strict=False)
        vocoder = vocoder.to(device="cpu").eval()
        logger.info(f"Vocoder loaded: {sum(p.numel() for p in vocoder.parameters())/1e6:.1f}M params")

        return audio_vae, vocoder

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, **kwargs) -> dict:
        return self._generate(**kwargs)

    def _ensure_denoising_ready(self):
        """Ensure scaffold and block streaming are set up on GPU.

        Keeps state persistent across generations — only the first call
        pays the setup cost. Subsequent calls are a no-op.
        """
        if self._streaming_state is not None:
            return

        self._move_transformer_scaffold_to_gpu()
        _log_gpu_memory("transformer scaffold on GPU")

        from scope_ltx_2.weight_streaming import (
            calculate_optimal_streaming_config,
            setup_block_streaming,
        )

        blocks = self._transformer.transformer_blocks
        free_mem, total_mem = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        available_gb = (total_mem - allocated) / 1024**3

        config = calculate_optimal_streaming_config(
            blocks,
            available_vram_gb=available_gb,
            safety_margin_gb=1.5,
            min_resident_blocks=4,
        )
        config.compute_device = self.device

        self._streaming_state = setup_block_streaming(blocks, config)
        _log_gpu_memory("block streaming setup")

    def _teardown_denoising(self):
        """Offload all blocks and scaffold to CPU to free GPU for other models."""
        self._cleanup_block_streaming()
        self._move_transformer_scaffold_to_cpu()
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _generate(self, **kwargs) -> dict:
        from .text_encoder import encode_prompt

        from .schema import VAE_SPATIAL_FACTOR, snap_frame_count, snap_to_multiple

        prompts = kwargs.get("prompts", [{"text": "a beautiful sunset", "weight": 1.0}])
        seed = kwargs.get("seed", kwargs.get("base_seed", 42))

        num_frames_raw = kwargs.get("num_frames", self.num_frames)
        num_frames = snap_frame_count(num_frames_raw)
        if num_frames != num_frames_raw:
            logger.info(f"Snapped num_frames {num_frames_raw} -> {num_frames} (must be N*8+1)")

        frame_rate = kwargs.get("frame_rate", self.frame_rate)

        height_raw = kwargs.get("height", self.height)
        height = snap_to_multiple(height_raw, VAE_SPATIAL_FACTOR)
        if height != height_raw:
            logger.info(f"Snapped height {height_raw} -> {height} (must be multiple of {VAE_SPATIAL_FACTOR})")

        width_raw = kwargs.get("width", self.width)
        width = snap_to_multiple(width_raw, VAE_SPATIAL_FACTOR)
        if width != width_raw:
            logger.info(f"Snapped width {width_raw} -> {width} (must be multiple of {VAE_SPATIAL_FACTOR})")

        randomize_seed = kwargs.get("randomize_seed", self.randomize_seed)

        if randomize_seed:
            seed = random.randint(0, 2**31 - 1)
            logger.info(f"Randomized seed: {seed}")

        prompt_text = prompts[0]["text"] if prompts else "a beautiful sunset"
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # =================================================================
        # Text encoding (Gemma FP8 on GPU -> offload -> connectors)
        # =================================================================
        if self._cached_prompt_text == prompt_text and self._cached_context is not None:
            context = self._cached_context
            logger.debug("Reusing cached prompt encoding")
        else:
            logger.info(f"Encoding prompt: {prompt_text[:80]}...")

            # Gemma needs ~13 GB — free everything else first
            self._teardown_denoising()
            self._offload_vaes()

            self._load_text_encoder()

            all_layer_hiddens = encode_prompt(
                self._text_encoder,
                self._tokenizer,
                prompt_text,
                device=self.device,
                dtype=self.dtype,
            )
            logger.info(f"Gemma encoding done: {all_layer_hiddens.shape}")

            self._offload_text_encoder()

            self._text_projection.to(self.device)
            projected = self._text_projection(all_layer_hiddens)
            self._text_projection.to("cpu")
            del all_layer_hiddens
            logger.info(f"Text projection done: {projected.shape}")

            self._move_connectors_to_gpu()
            context = self._transformer.preprocess_text_embeds(projected, unprocessed=True)
            self._move_connectors_to_cpu()

            del projected
            gc.collect()
            torch.cuda.empty_cache()

            self._cached_prompt_text = prompt_text
            self._cached_context = context
            _log_gpu_memory("text encoding complete")

        # Move VAEs to GPU before denoising setup so the streaming config
        # accounts for their memory in the available VRAM budget.
        self._move_vaes_to_gpu()

        # =================================================================
        # Prepare latent noise
        # =================================================================
        from .schema import VAE_TEMPORAL_FACTOR

        latent_frames = (num_frames - 1) // VAE_TEMPORAL_FACTOR + 1
        latent_height = height // VAE_SPATIAL_FACTOR
        latent_width = width // VAE_SPATIAL_FACTOR

        video_latents = torch.randn(
            (1, 128, latent_frames, latent_height, latent_width),
            generator=generator, device=self.device, dtype=self.dtype,
        )

        video_duration_sec = num_frames / frame_rate
        audio_latent_time = round(video_duration_sec * 25)
        audio_latents = torch.randn(
            (1, 8, audio_latent_time, 16),
            generator=generator, device=self.device, dtype=self.dtype,
        )

        # =================================================================
        # 8-step Euler denoising
        # =================================================================
        sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=self.device, dtype=self.dtype)
        logger.info(f"Denoising {num_frames} frames at {height}x{width} ({len(sigmas)-1} steps)")

        self._ensure_denoising_ready()
        self._reload_resident_blocks()

        video_latents, audio_latents = self._euler_denoise(
            video_latents, audio_latents, context, sigmas, frame_rate,
        )

        # Offload resident blocks to free VRAM for VAE decode.
        # Streaming state (hooks, pinned memory, scaffold) persists — resident
        # blocks will be reloaded by pre-forward hooks on the next denoising pass.
        self._free_blocks_for_decode()

        # =================================================================
        # VAE decode
        # =================================================================
        logger.info("Decoding video from latents...")
        video_tensor = self._decode_video(video_latents)

        logger.info("Decoding audio from latents...")
        audio_tensor, audio_sample_rate = self._decode_audio(audio_latents)

        logger.info(
            f"Generated: video={video_tensor.shape}, audio={audio_tensor.shape}, "
            f"duration={audio_tensor.shape[-1] / audio_sample_rate:.2f}s @ {audio_sample_rate}Hz"
        )

        return {
            "video": video_tensor,
            "audio": audio_tensor,
            "audio_sample_rate": audio_sample_rate,
            "frame_rate": frame_rate,
        }

    # ------------------------------------------------------------------
    # Transformer GPU streaming
    # ------------------------------------------------------------------

    def _move_connectors_to_gpu(self):
        """Move only the embedding connectors + caption projections to GPU."""
        t = self._transformer
        for attr in ("video_embeddings_connector", "audio_embeddings_connector",
                      "caption_projection", "audio_caption_projection"):
            mod = getattr(t, attr, None)
            if mod is not None and isinstance(mod, torch.nn.Module):
                _move_module_to(mod, self.device)
        _log_gpu_memory("connectors on GPU")

    def _move_connectors_to_cpu(self):
        t = self._transformer
        for attr in ("video_embeddings_connector", "audio_embeddings_connector",
                      "caption_projection", "audio_caption_projection"):
            mod = getattr(t, attr, None)
            if mod is not None and isinstance(mod, torch.nn.Module):
                _move_module_to(mod, "cpu")
        gc.collect()
        torch.cuda.empty_cache()

    _SCAFFOLD_EXCLUDE = {
        "transformer_blocks",
        "video_embeddings_connector", "audio_embeddings_connector",
        "caption_projection", "audio_caption_projection",
    }

    def _move_transformer_scaffold_to_gpu(self):
        """Move denoising-critical scaffold to GPU.

        Excludes transformer_blocks (managed by block streaming) and
        embedding connectors (only needed during text preprocessing,
        managed separately by _move_connectors_to_gpu/cpu).
        """
        t = self._transformer
        for name, child in t.named_children():
            if name not in self._SCAFFOLD_EXCLUDE:
                _move_module_to(child, self.device)
        for name, param in t.named_parameters(recurse=False):
            param.data = param.data.to(device=self.device)
        for name, buf in t.named_buffers(recurse=False):
            buf.data = buf.data.to(device=self.device)

    def _move_transformer_scaffold_to_cpu(self):
        t = self._transformer
        for name, child in t.named_children():
            if name not in self._SCAFFOLD_EXCLUDE:
                _move_module_to(child, "cpu")
        for name, param in t.named_parameters(recurse=False):
            param.data = param.data.to(device="cpu")
        for name, buf in t.named_buffers(recurse=False):
            buf.data = buf.data.to(device="cpu")

    def _cleanup_block_streaming(self):
        """Clean up block streaming and move all blocks back to CPU."""
        if self._streaming_state is None:
            return
        from scope_ltx_2.weight_streaming import cleanup_block_streaming
        cleanup_block_streaming(
            self._transformer.transformer_blocks,
            self._streaming_state,
            move_to="cpu",
        )
        self._streaming_state = None

    def _reload_resident_blocks(self):
        """Reload resident blocks to GPU after they were offloaded for VAE decode.

        Called before denoising to ensure all resident blocks are on GPU.
        Uses pinned memory from _free_blocks_for_decode for fast transfers.
        """
        if self._streaming_state is None:
            return
        state = self._streaming_state
        device = state.config.compute_device
        reloaded = 0
        for i in range(state.streaming_start_idx):
            if state.block_on_gpu[i]:
                continue
            block = state.blocks[i]
            for param in block.parameters():
                if param.device != device:
                    param.data = param.data.to(device, non_blocking=True)
            for buf in block.buffers():
                if buf.device != device:
                    buf.data = buf.data.to(device, non_blocking=True)
            state.block_on_gpu[i] = True
            reloaded += 1
        if reloaded > 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            _log_gpu_memory(f"reloaded {reloaded} resident blocks")

    def _free_blocks_for_decode(self):
        """Offload resident transformer blocks to free VRAM for VAE decode.

        Streaming state (hooks, pinned memory) remains intact. Resident
        blocks are pinned on first call for fast async transfers;
        subsequent calls just restore the pinned CPU pointers.
        _reload_resident_blocks() loads them back before the next denoising pass.
        """
        if self._streaming_state is None:
            return
        state = self._streaming_state
        offloaded = 0
        for i in range(state.streaming_start_idx):
            if not state.block_on_gpu[i]:
                continue
            if state._pinned_params[i]:
                block = state.blocks[i]
                for name, param in block.named_parameters():
                    if name in state._pinned_params[i]:
                        param.data = state._pinned_params[i][name]
                for buf in block.buffers():
                    if buf.device.type != "cpu":
                        buf.data = buf.data.to("cpu", non_blocking=True)
            else:
                state.pin_block(i)
            state.block_on_gpu[i] = False
            offloaded += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        _log_gpu_memory(f"{offloaded} resident blocks offloaded for VAE decode")

    def _euler_denoise(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        sigmas: torch.Tensor,
        frame_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """8-step Euler sampling with weight-streamed transformer blocks."""
        v_noisy = video_latents * sigmas[0]
        a_noisy = audio_latents * sigmas[0]

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            t0 = time.time()

            timestep = sigma.expand(v_noisy.shape[0])
            a_timestep = sigma.expand(a_noisy.shape[0])

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                model_output = self._transformer(
                    x=[v_noisy, a_noisy],
                    timestep=[timestep, a_timestep],
                    context=context,
                    attention_mask=None,
                    frame_rate=frame_rate,
                )

            if isinstance(model_output, list):
                v_pred = model_output[0]
                a_pred = model_output[1] if len(model_output) > 1 else a_noisy
            else:
                v_pred = model_output
                a_pred = a_noisy

            dt = sigma_next - sigma
            v_noisy = v_noisy + v_pred * dt
            a_noisy = a_noisy + a_pred * dt

            elapsed = time.time() - t0
            logger.info(
                f"Step {i+1}/{len(sigmas)-1}: sigma={sigma:.4f}->{sigma_next:.4f} ({elapsed:.1f}s) | "
                f"v_pred: mean={v_pred.mean().item():.4f} std={v_pred.std().item():.4f} | "
                f"v_noisy: mean={v_noisy.mean().item():.4f} std={v_noisy.std().item():.4f} range=[{v_noisy.min().item():.2f},{v_noisy.max().item():.2f}] | "
                f"a_pred: mean={a_pred.mean().item():.4f} std={a_pred.std().item():.4f} nan={a_pred.isnan().any().item()} | "
                f"a_noisy: mean={a_noisy.mean().item():.4f} std={a_noisy.std().item():.4f} range=[{a_noisy.min().item():.2f},{a_noisy.max().item():.2f}]"
            )

        return v_noisy, a_noisy

    # ------------------------------------------------------------------
    # VAE decode
    # ------------------------------------------------------------------

    def _decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode video latents using ComfyUI-compatible VideoVAE.

        VideoVAE.decode() handles per-channel un-normalization internally.
        Output is in [-1, 1] range, we convert to [0, 1] and rearrange to (F, H, W, C).
        """
        need_move = not self._vaes_on_gpu
        if need_move:
            _move_module_to(self._video_vae, self.device)
        try:
            logger.info(f"VAE input: shape={latents.shape} mean={latents.mean().item():.4f} std={latents.std().item():.4f} range=[{latents.min().item():.2f},{latents.max().item():.2f}]")
            decoded = self._video_vae.decode(latents)
            decoded = decoded.to(torch.float32)
            logger.info(f"VAE decoded: shape={decoded.shape} mean={decoded.mean().item():.4f} std={decoded.std().item():.4f} range=[{decoded.min().item():.2f},{decoded.max().item():.2f}]")
            video = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
            video = video.squeeze(0)
            video = video.permute(1, 2, 3, 0)
            logger.info(f"Video output: shape={video.shape} mean={video.mean().item():.4f} range=[{video.min().item():.2f},{video.max().item():.2f}]")
        finally:
            if need_move:
                _move_module_to(self._video_vae, "cpu")
                gc.collect()
                torch.cuda.empty_cache()
        return video

    def _decode_audio(self, latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Decode audio latents following ComfyUI's AudioVAE.decode() pattern.

        Returns (waveform, sample_rate) where sample_rate is the vocoder's
        actual output rate (with BWE this is typically 48 kHz, not the base
        VAE's 24 kHz).

        Steps (matching ComfyUI AudioVAE.decode + run_vocoder):
        1. Denormalize: patchify -> un_normalize -> unpatchify
        2. Compute target_shape from original latent dimensions
        3. Decoder forward -> mel spectrogram
        4. Transpose mel for vocoder, squeeze if mono
        5. Vocoder -> waveform
        """
        from einops import rearrange

        LATENT_DOWNSAMPLE_FACTOR = 4

        need_move = not self._vaes_on_gpu
        if need_move:
            _move_module_to(self._audio_vae, self.device)
            _move_module_to(self._vocoder, self.device)
        try:
            # Float32 for signal-processing precision (STFT, mel filterbank)
            latents = latents.to(device=self.device, dtype=torch.float32)

            original_shape = latents.shape
            batch, channels, time, freq = original_shape
            logger.info(
                f"Audio decode input: shape={original_shape} "
                f"mean={latents.mean().item():.4f} std={latents.std().item():.4f} "
                f"range=[{latents.min().item():.2f},{latents.max().item():.2f}]"
            )

            # Denormalize: patchify -> un_normalize -> unpatchify
            patched = rearrange(latents, "b c t f -> b t (c f)")
            denormalized = self._audio_vae.per_channel_statistics.un_normalize(patched)
            latents = rearrange(denormalized, "b t (c f) -> b c t f", c=channels, f=freq)

            # Target shape (ComfyUI AudioVAE.target_shape_from_latents)
            from .comfy_vae.audio_vae import CausalityAxis
            target_length = time * LATENT_DOWNSAMPLE_FACTOR
            if self._audio_vae.causality_axis != CausalityAxis.NONE:
                target_length -= LATENT_DOWNSAMPLE_FACTOR - 1
            target_shape = (
                batch,
                self._audio_vae.decoder.out_ch,
                target_length,
                self._audio_vae.mel_bins,
            )

            mel_spec = self._audio_vae.decode(latents, target_shape=target_shape)
            logger.info(
                f"Audio mel_spec: shape={mel_spec.shape} "
                f"mean={mel_spec.mean().item():.4f} std={mel_spec.std().item():.4f}"
            )

            # Run vocoder (ComfyUI AudioVAE.run_vocoder)
            audio_channels = self._audio_vae.decoder.out_ch
            vocoder_input = mel_spec.transpose(2, 3)
            if audio_channels == 1:
                vocoder_input = vocoder_input.squeeze(1)
            elif audio_channels != 2:
                raise ValueError(f"Unsupported audio_channels: {audio_channels}")

            waveform = self._vocoder(vocoder_input)
            logger.info(
                f"Audio waveform: shape={waveform.shape} "
                f"mean={waveform.mean().item():.4f} std={waveform.std().item():.4f} "
                f"range=[{waveform.min().item():.2f},{waveform.max().item():.2f}]"
            )

            audio = waveform.squeeze(0).float()
        finally:
            if need_move:
                _move_module_to(self._audio_vae, "cpu")
                _move_module_to(self._vocoder, "cpu")
                gc.collect()
                torch.cuda.empty_cache()

        # Output sample rate from vocoder (ComfyUI AudioVAE.output_sample_rate)
        output_rate = getattr(self._vocoder, "output_sample_rate", None)
        if output_rate is not None:
            sample_rate = int(output_rate)
        else:
            upsample_factor = getattr(self._vocoder, "upsample_factor", None)
            if upsample_factor is not None:
                sample_rate = int(
                    self._audio_vae.sampling_rate * upsample_factor / self._audio_vae.mel_hop_length
                )
            else:
                sample_rate = int(self._audio_vae.sampling_rate)
        logger.info(f"Audio output: {audio.shape}, sample_rate={sample_rate}Hz")
        return audio, sample_rate


class _ChunkedFFN(torch.nn.Module):
    def __init__(self, ff: torch.nn.Module, chunk_size: int):
        super().__init__()
        self.ff = ff
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= self.chunk_size:
            return self.ff(x)
        chunks = []
        for i in range(0, x.shape[1], self.chunk_size):
            chunks.append(self.ff(x[:, i:i + self.chunk_size, :]))
        return torch.cat(chunks, dim=1)
