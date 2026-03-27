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


def _dilate_latent(
    samples: torch.Tensor,
    horizontal_scale: int,
    vertical_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spread a small latent into a larger spatial grid with a validity mask.

    Ported from ComfyUI-LTXVideo ``LTXVDilateLatent``.  Given a latent of
    shape ``(B, C, F, H, W)`` the output has spatial dimensions
    ``(H * vertical_scale, W * horizontal_scale)`` with the original values
    placed at stride positions and zeros elsewhere.  The returned mask is
    ``1.0`` at valid (stride) positions and ``-1.0`` at padding positions;
    downstream code uses the negative sentinel to filter those tokens out.

    Returns:
        (dilated_samples, dilated_mask) — both ``(B, *, F, H_out, W_out)``.
    """
    if horizontal_scale == 1 and vertical_scale == 1:
        mask = torch.ones(
            (samples.shape[0], 1, samples.shape[2], samples.shape[3], samples.shape[4]),
            device=samples.device, dtype=samples.dtype,
        )
        return samples, mask

    dilated_shape = samples.shape[:3] + (
        samples.shape[3] * vertical_scale,
        samples.shape[4] * horizontal_scale,
    )
    dilated_samples = torch.zeros(dilated_shape, device=samples.device, dtype=samples.dtype)
    dilated_samples[..., ::vertical_scale, ::horizontal_scale] = samples

    dilated_mask = torch.full(
        (dilated_samples.shape[0], 1, dilated_samples.shape[2],
         dilated_samples.shape[3], dilated_samples.shape[4]),
        -1.0, device=samples.device, dtype=samples.dtype,
    )
    dilated_mask[..., ::vertical_scale, ::horizontal_scale] = 1.0
    return dilated_samples, dilated_mask


def _video_input_to_frames(
    video_input: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert scope video input to float frame tensor.

    Scope pipelines receive video as a list of ``(1, H, W, C)`` uint8
    tensors via the ``"video"`` graph port.  This converts to the
    ``(F, H, W, C)`` float ``[0, 1]`` format used by ``_encode_image``.
    """
    stacked = torch.cat(video_input, dim=0)          # (F, H, W, C)
    return stacked.to(device=device, dtype=dtype).float() / 255.0


def _load_image_tensor(source, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Load an image from a file path or pass through an existing tensor.

    Args:
        source: File path (str/Path) or tensor in ``(H, W, C)``/``(F, H, W, C)`` [0, 1].
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape ``(F, H, W, C)`` in ``[0, 1]`` float range on *device*.
    """
    if isinstance(source, (str, Path)):
        from PIL import Image
        import torchvision.transforms.functional as TF

        img = Image.open(str(source)).convert("RGB")
        # to_tensor returns (C, H, W) in [0, 1]
        tensor = TF.to_tensor(img).permute(1, 2, 0)  # -> (H, W, C)
        return tensor.unsqueeze(0).to(device=device, dtype=dtype)  # -> (1, H, W, C)

    t = source.to(device=device, dtype=dtype)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t


class LTX2Pipeline(Pipeline):
    """LTX 2.3 audio-video generation pipeline."""

    _current_instance: "LTX2Pipeline | None" = None

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LTX2Config

    def prepare(self, **kwargs):
        """Return input requirements based on current mode.

        When video input is active (depth/control frames via graph port),
        requests ``num_frames`` frames so the full control sequence is
        available before generation starts.
        """
        from scope.core.pipelines.interface import Requirements

        if kwargs.get("video") is not None:
            num_frames = kwargs.get("num_frames", self.num_frames)
            return Requirements(input_size=num_frames)
        return None

    def __init__(
        self,
        height: int = 384,
        width: int = 320,
        num_frames: int = 129,
        frame_rate: float = 24.0,
        randomize_seed: bool = True,
        num_steps: int = 8,
        schedule: str = "distilled",
        sigmas: list[float] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        transformer_path: str | None = None,
        text_projection_path: str | None = None,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        gemma_model_path: str | None = None,
        gemma_tokenizer_path: str | None = None,
        ffn_chunk_size: int | None = 4096,
        loras: list[dict] | None = None,
        **kwargs,
    ):
        from .model_loader import load_transformer
        from .schema import make_sigma_schedule
        from .text_encoder import TextEmbeddingProjection, load_gemma_text_encoder

        prev = LTX2Pipeline._current_instance
        if prev is not None:
            logger.info("Cleaning up previous LTX2Pipeline instance before loading new one...")
            prev.unload()
            LTX2Pipeline._current_instance = None

        self._cancelled = False

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.randomize_seed = randomize_seed
        self.num_steps = num_steps
        self.schedule = schedule
        self.sigmas = sigmas if sigmas is not None else make_sigma_schedule(num_steps, schedule)
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

        # Step 3b: Merge user LoRA weights (before FFN chunking / streaming setup)
        if loras is None:
            loras = []
        if loras:
            from .lora import load_and_merge_loras
            self.loaded_lora_adapters = load_and_merge_loras(self._transformer, loras)
        else:
            self.loaded_lora_adapters = []

        # Step 3c: Detect IC-LoRA Union Control for deferred merge.
        # IC-LoRA is NOT merged at init because its weight modifications
        # interfere with user LoRAs (e.g. style LoRAs) in text-only mode.
        # It will be merged on first use when video guide input is received.
        IC_LORA_FILENAME = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
        ic_lora_path = get_model_file_path(
            f"LTX-2.3-22b-IC-LoRA-Union-Control/{IC_LORA_FILENAME}"
        )
        already_has_ic_lora = any(
            IC_LORA_FILENAME in (cfg.get("path") or "") for cfg in loras
        )
        if ic_lora_path.exists() and not already_has_ic_lora:
            self._pending_ic_lora = {"path": str(ic_lora_path), "scale": 1.0}
            logger.info(f"IC-LoRA Union Control available (deferred): {ic_lora_path}")
        else:
            self._pending_ic_lora = None

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

        LTX2Pipeline._current_instance = self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload(self):
        """Release all GPU resources for a clean pipeline restart.

        Called automatically when a new LTX2Pipeline is created, and as a
        safety net from __del__. Sets the cancellation flag so any in-flight
        generation aborts at the next denoising step.
        """
        self._cancelled = True

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        try:
            self._cleanup_block_streaming()
        except Exception as e:
            logger.warning(f"Block streaming cleanup error during unload: {e}")

        try:
            self._move_transformer_scaffold_to_cpu()
        except Exception as e:
            logger.warning(f"Scaffold offload error during unload: {e}")

        try:
            self._move_connectors_to_cpu()
        except Exception as e:
            logger.warning(f"Connector offload error during unload: {e}")

        try:
            if getattr(self, '_text_encoder_on_gpu', False):
                _move_module_to(self._text_encoder, "cpu")
                self._text_encoder_on_gpu = False
        except Exception as e:
            logger.warning(f"Text encoder offload error during unload: {e}")

        try:
            if getattr(self, '_vaes_on_gpu', False):
                for attr in ('_video_vae', '_audio_vae', '_vocoder'):
                    mod = getattr(self, attr, None)
                    if mod is not None:
                        _move_module_to(mod, "cpu")
                self._vaes_on_gpu = False
        except Exception as e:
            logger.warning(f"VAE offload error during unload: {e}")

        for attr in ('_text_encoder', '_tokenizer', '_text_projection',
                      '_transformer', '_video_vae', '_audio_vae', '_vocoder'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

        self._cached_context = None
        self._cached_prompt_text = None
        self._streaming_state = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _log_gpu_memory("pipeline unloaded")

        if LTX2Pipeline._current_instance is self:
            LTX2Pipeline._current_instance = None

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model management helpers
    # ------------------------------------------------------------------

    def _merge_ic_lora(self):
        """Merge the deferred IC-LoRA Union Control into the transformer.

        Called on first use of video guide conditioning. Requires the
        transformer to be CPU-resident (before streaming setup) and
        tears down any existing denoising state first.
        """
        if self._pending_ic_lora is None:
            return

        from .lora import load_and_merge_loras

        self._teardown_denoising()

        ic_cfg = self._pending_ic_lora
        self._pending_ic_lora = None
        logger.info(f"Merging deferred IC-LoRA: {Path(ic_cfg['path']).name}")
        merged = load_and_merge_loras(self._transformer, [ic_cfg])
        self.loaded_lora_adapters.extend(merged)

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

    def _ensure_denoising_ready(self, total_tokens: int = 0):
        """Ensure scaffold and block streaming are set up on GPU.

        Keeps state persistent across generations — only the first call
        pays the setup cost. Subsequent calls are a no-op.

        Args:
            total_tokens: Total sequence length (video + guide tokens) so
                the safety margin can be scaled for self-attention memory.
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

        # Forward pass intermediates (Q/K/V projections, FFN activations, etc.)
        # scale linearly with sequence length. Base safety of 1.5 GB is
        # calibrated for the normal ~2k token case; scale up for IC-LoRA.
        BASE_TOKENS = 2040
        safety_gb = 1.5
        if total_tokens > BASE_TOKENS:
            ratio = total_tokens / BASE_TOKENS
            safety_gb = 1.5 * ratio
            logger.info(
                f"Streaming safety margin: {safety_gb:.1f} GB "
                f"({total_tokens} tokens, {ratio:.1f}x base)"
            )

        config = calculate_optimal_streaming_config(
            blocks,
            available_vram_gb=available_gb,
            safety_margin_gb=safety_gb,
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

        from .schema import VAE_SPATIAL_FACTOR, make_sigma_schedule, snap_frame_count, snap_to_multiple

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
        # Prepare sigma schedule (needed before latent init for i2v)
        # =================================================================
        custom_sigmas = kwargs.get("sigmas")
        if custom_sigmas is not None:
            sigma_values = custom_sigmas
        else:
            num_steps = kwargs.get("num_steps", self.num_steps)
            schedule = kwargs.get("schedule", self.schedule)
            sigma_values = make_sigma_schedule(num_steps, schedule)
        sigmas = torch.tensor(sigma_values, device=self.device, dtype=self.dtype)

        # =================================================================
        # Prepare latent noise + optional i2v conditioning
        # =================================================================
        from .schema import VAE_TEMPORAL_FACTOR

        latent_frames = (num_frames - 1) // VAE_TEMPORAL_FACTOR + 1
        latent_height = height // VAE_SPATIAL_FACTOR
        latent_width = width // VAE_SPATIAL_FACTOR

        video_noise = torch.randn(
            (1, 128, latent_frames, latent_height, latent_width),
            generator=generator, device=self.device, dtype=self.dtype,
        )

        # Check for i2v image input (file path or tensor)
        i2v_source = kwargs.get("i2v_image") or kwargs.get("first_frame_image")
        i2v_strength = float(kwargs.get("i2v_strength", 1.0))
        denoise_mask = None
        clean_latent = None

        if i2v_source is not None:
            logger.info(f"Image-to-video conditioning (strength={i2v_strength})")
            i2v_image = _load_image_tensor(i2v_source, self.device, self.dtype)
            image_latent = self._encode_image(i2v_image, height, width)
            cond_frames = image_latent.shape[2]
            logger.info(
                f"Encoded i2v image: {i2v_image.shape} -> latent {image_latent.shape} "
                f"({cond_frames} conditioned frame(s))"
            )

            clean_latent = torch.zeros_like(video_noise)
            clean_latent[:, :, :cond_frames] = image_latent

            # denoise_mask: 0 = fully conditioned, 1 = fully noisy
            denoise_mask = torch.ones(
                (1, 1, latent_frames, 1, 1),
                device=self.device, dtype=self.dtype,
            )
            denoise_mask[:, :, :cond_frames] = 1.0 - i2v_strength

        # CONST noise_scaling at sigma_0: sigma*noise + (1-sigma)*latent_image.
        # At sigma=1 this is pure noise regardless of i2v — the conditioning
        # is injected via per-step masking inside _euler_denoise, not here.
        video_latents = video_noise

        # =================================================================
        # IC-LoRA guide conditioning (depth / canny / pose via video port)
        # =================================================================
        video_input = kwargs.get("video")
        control_strength = float(kwargs.get("control_strength", 1.0))
        keyframe_idxs = None
        num_guide_latent_frames = 0

        has_ic_lora = any(
            a.get("reference_downscale_factor", 1.0) > 1.0
            for a in self.loaded_lora_adapters
        )
        ic_lora_available = has_ic_lora or self._pending_ic_lora is not None

        if video_input is not None and ic_lora_available:
            if not has_ic_lora:
                self._merge_ic_lora()
            logger.info(f"IC-LoRA guide conditioning (strength={control_strength})")
            control_frames = _video_input_to_frames(video_input, self.device, self.dtype)
            logger.info(f"Control frames: {control_frames.shape[0]} frames {control_frames.shape}")

            (video_latents, denoise_mask, clean_latent,
             keyframe_idxs, num_guide_latent_frames) = self._prepare_ic_lora_guide(
                control_frames, video_latents, denoise_mask, clean_latent,
                height, width, control_strength,
            )
        elif video_input is not None:
            logger.warning(
                "Video input received but no IC-LoRA Union Control loaded — "
                "skipping guide conditioning (output would be corrupted). "
                "Falling back to text-only generation."
            )

        video_duration_sec = num_frames / frame_rate
        audio_latent_time = round(video_duration_sec * 25)
        audio_latents = torch.randn(
            (1, 8, audio_latent_time, 16),
            generator=generator, device=self.device, dtype=self.dtype,
        )

        # =================================================================
        # Euler denoising
        # =================================================================
        logger.info(f"Denoising {num_frames} frames at {height}x{width} ({len(sigmas)-1} steps)")

        total_tokens = video_latents.shape[2] * video_latents.shape[3] * video_latents.shape[4]
        self._ensure_denoising_ready(total_tokens=total_tokens)
        self._reload_resident_blocks()

        video_latents, audio_latents = self._euler_denoise(
            video_latents, audio_latents, context, sigmas, frame_rate,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
            keyframe_idxs=keyframe_idxs,
        )

        # Strip appended guide frames before VAE decode
        if num_guide_latent_frames > 0:
            video_latents = video_latents[:, :, :-num_guide_latent_frames]
            logger.info(
                f"Stripped {num_guide_latent_frames} guide frames, "
                f"video_latents now {video_latents.shape}"
            )

        if self._cancelled:
            logger.info("Generation cancelled, skipping VAE decode")
            return {
                "video": torch.zeros(1, height, width, 3),
                "audio": torch.zeros(2, 1),
                "audio_sample_rate": 48000,
                "frame_rate": frame_rate,
            }

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
        denoise_mask: torch.Tensor | None = None,
        clean_latent: torch.Tensor | None = None,
        keyframe_idxs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler sampling with weight-streamed transformer blocks.

        For i2v / IC-LoRA conditioning this replicates ComfyUI's
        ``KSamplerX0Inpaint`` + ``LTXAV.process_timestep`` behaviour:

        1. **Pre-model masking** — conditioned frames are replaced with the
           clean ``clean_latent`` (LTXAV ``scale_latent_inpaint`` returns
           ``latent_image`` directly, no noise).
        2. **Per-patch timestep** — ``denoise_mask * sigma`` is patchified so
           conditioned patches receive ``timestep=0`` (model AdaLN treats
           them as clean) while unconditioned patches receive the current
           ``sigma``.
        3. **Post-step masking** — after the Euler update, conditioned frames
           are anchored back to ``clean_latent`` (equivalent to ComfyUI's
           post-model output masking which forces ``d=0`` for those frames).

        When ``keyframe_idxs`` is provided (IC-LoRA guide conditioning), the
        transformer's ``_process_input`` uses negative ``denoise_mask`` values
        to filter padding tokens from dilated guide latents and injects the
        custom RoPE coordinates for the guide tokens.
        """
        v_noisy = video_latents * sigmas[0]
        a_noisy = audio_latents * sigmas[0]

        for i in range(len(sigmas) - 1):
            if self._cancelled:
                logger.info("Denoising cancelled, aborting")
                break

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            t0 = time.time()

            # -- pre-model masking: lock conditioned frames to clean values
            if denoise_mask is not None and clean_latent is not None:
                v_noisy = v_noisy * denoise_mask + clean_latent * (1.0 - denoise_mask)

            # -- build per-patch video timestep (ComfyUI LTXAV.process_timestep)
            if denoise_mask is not None:
                _, _, lf, lh, lw = v_noisy.shape
                mask_vol = denoise_mask.expand(1, 1, lf, lh, lw)
                per_voxel_sigma = mask_vol * sigma          # 0 for conditioned, sigma for others
                v_timestep = self._transformer.patchifier.patchify(per_voxel_sigma)[0]
                v_timestep = v_timestep.squeeze(-1)         # (1, num_patches)
            else:
                v_timestep = sigma.expand(v_noisy.shape[0])

            a_timestep = sigma.expand(a_noisy.shape[0])

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                model_output = self._transformer(
                    x=[v_noisy, a_noisy],
                    timestep=[v_timestep, a_timestep],
                    context=context,
                    attention_mask=None,
                    frame_rate=frame_rate,
                    keyframe_idxs=keyframe_idxs,
                    denoise_mask=denoise_mask,
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

            # -- i2v post-step masking: re-anchor conditioned frames
            if denoise_mask is not None and clean_latent is not None:
                v_noisy = v_noisy * denoise_mask + clean_latent * (1.0 - denoise_mask)

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
    # VAE encode (for i2v / IC-LoRA guide conditioning)
    # ------------------------------------------------------------------

    def _encode_image(
        self,
        image: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        """Encode image(s) into video latent space using the VideoVAE encoder.

        Args:
            image: Tensor of shape ``(F, H, W, C)`` in ``[0, 1]`` float range,
                   or ``(H, W, C)`` for a single frame.
            target_height: Desired pixel height (must be multiple of 32).
            target_width: Desired pixel width (must be multiple of 32).

        Returns:
            Latent tensor of shape ``(1, 128, F_lat, H_lat, W_lat)`` where
            ``F_lat = (F-1)//8 + 1``, ``H_lat = target_height//32``,
            ``W_lat = target_width//32``.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)  # (H, W, C) -> (1, H, W, C)

        # (F, H, W, C) -> (1, C, F, H, W)
        pixels = image.permute(3, 0, 1, 2).unsqueeze(0)  # C,F,H,W -> 1,C,F,H,W

        # Resize to target resolution
        if pixels.shape[3] != target_height or pixels.shape[4] != target_width:
            from einops import rearrange
            b, c, f, h, w = pixels.shape
            frames_flat = rearrange(pixels, "b c f h w -> (b f) c h w")
            frames_flat = torch.nn.functional.interpolate(
                frames_flat, size=(target_height, target_width), mode="bilinear", align_corners=False,
            )
            pixels = rearrange(frames_flat, "(b f) c h w -> b c f h w", b=b, f=f)

        # [0, 1] -> [-1, 1]
        pixels = pixels * 2.0 - 1.0
        pixels = pixels[:, :3]  # keep only RGB

        need_move = not self._vaes_on_gpu
        if need_move:
            _move_module_to(self._video_vae, self.device)
        try:
            pixels = pixels.to(device=self.device, dtype=self.dtype)
            latent = self._video_vae.encode(pixels)
            logger.info(
                f"VAE encode: input={pixels.shape} -> latent={latent.shape} "
                f"mean={latent.mean().item():.4f} std={latent.std().item():.4f}"
            )
        finally:
            if need_move:
                _move_module_to(self._video_vae, "cpu")
                gc.collect()
                torch.cuda.empty_cache()
        return latent

    # ------------------------------------------------------------------
    # IC-LoRA guide conditioning
    # ------------------------------------------------------------------

    def _prepare_ic_lora_guide(
        self,
        control_frames: torch.Tensor,
        video_latents: torch.Tensor,
        denoise_mask: torch.Tensor | None,
        clean_latent: torch.Tensor | None,
        height: int,
        width: int,
        strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Prepare IC-LoRA guide conditioning for depth/control-driven generation.

        Replicates ComfyUI's ``LTXAddVideoICLoRAGuide`` workflow:

        1. Encode control frames at reduced resolution (target / downscale_factor)
        2. Dilate the small latent into a full-resolution grid with validity mask
        3. Append dilated guide to video latents along temporal dimension
        4. Build ``keyframe_idxs`` with adjusted RoPE coordinates
        5. Combine masks for the Euler denoising loop

        Args:
            control_frames: ``(F, H, W, C)`` in ``[0, 1]``.
            video_latents: ``(1, 128, F_lat, H_lat, W_lat)`` noise latent.
            denoise_mask: Existing mask from i2v (or None).
            clean_latent: Existing clean latent from i2v (or None).
            height: Output pixel height.
            width: Output pixel width.
            strength: Guide conditioning strength (1.0 = fully conditioned).

        Returns:
            (video_latents, denoise_mask, clean_latent, keyframe_idxs, num_guide_frames)
        """
        from .patchifier import SymmetricPatchifier, latent_to_pixel_coords
        from .schema import VAE_SPATIAL_FACTOR, VAE_TEMPORAL_FACTOR

        downscale_factor = 1.0
        for adapter in self.loaded_lora_adapters:
            dsf = adapter.get("reference_downscale_factor", 1.0)
            if dsf > 1.0:
                downscale_factor = dsf
                break

        _, _, latent_frames, latent_height, latent_width = video_latents.shape
        scale_factors = (VAE_TEMPORAL_FACTOR, VAE_SPATIAL_FACTOR, VAE_SPATIAL_FACTOR)
        int_dsf = int(downscale_factor)

        # Trim control frames to valid VAE temporal alignment (N*8 + 1)
        n_raw = control_frames.shape[0]
        n_keep = ((n_raw - 1) // VAE_TEMPORAL_FACTOR) * VAE_TEMPORAL_FACTOR + 1
        causal_fix = True  # frame_idx == 0

        if not causal_fix:
            control_frames = torch.cat([control_frames[:1], control_frames], dim=0)
        control_frames = control_frames[:n_keep]

        # Encode at reduced resolution
        target_height = int(height / downscale_factor)
        target_width = int(width / downscale_factor)
        guide_latent = self._encode_image(control_frames, target_height, target_width)

        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]
            control_frames = control_frames[1:]

        logger.info(
            f"IC-LoRA guide: {n_raw} frames -> {n_keep} kept, "
            f"encoded at {target_height}x{target_width} -> latent {guide_latent.shape}, "
            f"downscale_factor={downscale_factor}"
        )

        guide_orig_shape = list(guide_latent.shape[2:])  # [F, H_small, W_small]

        # Dilate if downscale_factor > 1
        guide_mask = None
        if int_dsf > 1:
            if latent_width % int_dsf != 0 or latent_height % int_dsf != 0:
                raise ValueError(
                    f"Latent spatial size {latent_width}x{latent_height} must be "
                    f"divisible by latent_downscale_factor {int_dsf}"
                )
            guide_latent, guide_mask = _dilate_latent(guide_latent, int_dsf, int_dsf)
            logger.info(
                f"IC-LoRA dilated guide: {guide_orig_shape} -> {list(guide_latent.shape[2:])}"
            )

        num_guide_frames = guide_latent.shape[2]

        # Build keyframe_idxs: pixel coordinates for RoPE
        patchifier = SymmetricPatchifier(1, start_end=True)
        _, latent_coords = patchifier.patchify(guide_latent)
        pixel_coords = latent_to_pixel_coords(
            latent_coords, scale_factors, causal_fix=causal_fix,
        )
        pixel_coords[:, 0] += 0  # frame_idx = 0

        # Adjust end positions for the downscaled grid so RoPE encodes the
        # correct centre of each token's receptive field.
        if int_dsf > 1:
            spatial_end_offset = (int_dsf - 1) * torch.tensor(
                scale_factors[1:], device=pixel_coords.device,
            ).view(1, -1, 1, 1)
            pixel_coords[:, 1:, :, 1:] += spatial_end_offset.to(pixel_coords.dtype)

        keyframe_idxs = pixel_coords

        # Build guide mask entry for denoise_mask concatenation
        if guide_mask is not None:
            target_h = max(
                1 if denoise_mask is None else denoise_mask.shape[3],
                guide_mask.shape[3],
            )
            target_w = max(
                1 if denoise_mask is None else denoise_mask.shape[4],
                guide_mask.shape[4],
            )

            if denoise_mask is not None and (
                denoise_mask.shape[3] == 1 or denoise_mask.shape[4] == 1
            ):
                denoise_mask = denoise_mask.expand(-1, -1, -1, target_h, target_w).clone()

            if guide_mask.shape[3] == 1 or guide_mask.shape[4] == 1:
                guide_mask = guide_mask.expand(-1, -1, -1, target_h, target_w)

            mask_entry = guide_mask - strength
        else:
            spatial_h = 1 if denoise_mask is None else denoise_mask.shape[3]
            spatial_w = 1 if denoise_mask is None else denoise_mask.shape[4]
            mask_entry = torch.full(
                (1, 1, num_guide_frames, spatial_h, spatial_w),
                1.0 - strength,
                dtype=video_latents.dtype, device=video_latents.device,
            )

        # Initialise denoise_mask / clean_latent if not already set by i2v
        if denoise_mask is None:
            denoise_mask = torch.ones(
                (1, 1, latent_frames, 1, 1),
                device=video_latents.device, dtype=video_latents.dtype,
            )
        if clean_latent is None:
            clean_latent = torch.zeros_like(video_latents)

        # Expand existing denoise_mask spatially if mask_entry has spatial dims
        if mask_entry.shape[3] > 1 and denoise_mask.shape[3] == 1:
            denoise_mask = denoise_mask.expand(
                -1, -1, -1, mask_entry.shape[3], mask_entry.shape[4],
            ).clone()

        # Append guide to latents, mask, and clean_latent
        video_latents = torch.cat([video_latents, guide_latent], dim=2)
        denoise_mask = torch.cat([denoise_mask, mask_entry], dim=2)
        clean_latent = torch.cat([clean_latent, guide_latent], dim=2)

        # _process_input patchifies denoise_mask and x independently; both must
        # produce the same number of tokens.  Expand spatial dims to match.
        _, _, _, lat_h, lat_w = video_latents.shape
        if denoise_mask.shape[3] != lat_h or denoise_mask.shape[4] != lat_w:
            denoise_mask = denoise_mask.expand(-1, -1, -1, lat_h, lat_w).contiguous()

        logger.info(
            f"IC-LoRA guide appended: video_latents {video_latents.shape}, "
            f"denoise_mask {denoise_mask.shape}, "
            f"keyframe_idxs {keyframe_idxs.shape}, "
            f"num_guide_frames={num_guide_frames}"
        )

        return video_latents, denoise_mask, clean_latent, keyframe_idxs, num_guide_frames

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
