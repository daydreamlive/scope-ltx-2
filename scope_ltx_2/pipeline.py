"""LTX 2.3 audio-video generation pipeline for Daydream Scope.

Uses Kijai's separated ComfyUI-format checkpoints (distilled v3 FP8) with
a standalone inference implementation adapted from ComfyUI's lightricks code.

Memory Management:
    The 22B FP8 transformer (~23GB) cannot fit on a 24GB GPU alongside other
    models. The pipeline keeps the transformer on CPU and moves it to GPU
    only during denoising (after the text encoder is offloaded). This mirrors
    ComfyUI's --lowvram approach.

    Loading order:
    1. Gemma 3 12B → GPU → encode prompt → offload to CPU
    2. Text projection (aggregate embeds) → CPU (tiny, ~1.5GB)
    3. Transformer → CPU (23GB system RAM)
    4. VAEs → CPU (small)
    5. During inference: move transformer to GPU, denoise, move back

Audio Support:
    LTX 2.3 natively supports synchronized audio-video generation.
"""

import gc
import logging
import random
import time
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_model_file_path
from scope.core.pipelines.interface import Pipeline

from .schema import LTX2Config

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DISTILLED_SIGMA_VALUES = [1.0, 0.993, 0.955, 0.865, 0.698, 0.463, 0.216, 0.044, 0.0]

AUDIO_SAMPLE_RATE = 24000


def _log_gpu_memory(stage: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory after {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class LTX2Pipeline(Pipeline):
    """LTX 2.3 audio-video generation pipeline.

    Uses Kijai's separated checkpoints with ComfyUI-derived model code.
    Fits on a 24GB GPU via CPU offloading of the transformer and text encoder.
    """

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
        gemma_root: str | None = None,
        ffn_chunk_size: int | None = 4096,
        offload_text_encoder: bool = True,
        **kwargs,
    ):
        from pathlib import Path

        from .model_loader import load_transformer, load_vae
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
        self.offload_text_encoder = offload_text_encoder
        self.ffn_chunk_size = ffn_chunk_size

        if kwargs:
            logger.debug(f"LTX2Pipeline ignoring unknown kwargs: {list(kwargs.keys())}")

        start = time.time()

        # Resolve model paths
        kijai_dir = get_model_file_path("LTX2.3_comfy")

        if transformer_path is None:
            transformer_path = str(kijai_dir / "diffusion_models" / "ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors")
        if text_projection_path is None:
            text_projection_path = str(kijai_dir / "text_encoders" / "ltx-2.3_text_projection_bf16.safetensors")
        if video_vae_path is None:
            video_vae_path = str(kijai_dir / "vae" / "LTX23_video_vae_bf16.safetensors")
        if audio_vae_path is None:
            audio_vae_path = str(kijai_dir / "vae" / "LTX23_audio_vae_bf16.safetensors")
        if gemma_root is None:
            gemma_root = str(get_model_file_path("gemma-3-12b-it"))

        logger.info(f"Transformer: {transformer_path}")
        logger.info(f"Text projection: {text_projection_path}")
        logger.info(f"Video VAE: {video_vae_path}")
        logger.info(f"Audio VAE: {audio_vae_path}")
        logger.info(f"Gemma: {gemma_root}")

        # Step 1: Load text encoder to GPU, then offload
        logger.info("Loading Gemma 3 12B text encoder...")
        self._text_encoder, self._tokenizer = load_gemma_text_encoder(
            gemma_root, device=device, dtype=dtype
        )
        self._text_encoder_on_gpu = True
        _log_gpu_memory("text encoder loaded")

        if offload_text_encoder:
            logger.info("Offloading text encoder to CPU...")
            self._text_encoder.to("cpu")
            self._text_encoder_on_gpu = False
            gc.collect()
            torch.cuda.empty_cache()
            _log_gpu_memory("text encoder offloaded")

        # Step 2: Load text projection (aggregate embeds, ~1.5GB on CPU)
        logger.info("Loading text projection (aggregate embeds)...")
        self._text_projection = TextEmbeddingProjection.from_checkpoint(
            text_projection_path, dtype=dtype
        )
        self._text_projection_on_gpu = False

        # Step 3: Load transformer to CPU (23GB, too large for GPU alongside other models)
        logger.info("Loading transformer (FP8, will stay on CPU until inference)...")
        self._transformer = load_transformer(transformer_path, device=torch.device("cpu"), dtype=dtype)
        self._transformer_on_gpu = False

        if ffn_chunk_size is not None:
            self._apply_ffn_chunking(ffn_chunk_size)

        _log_gpu_memory("transformer loaded (on CPU)")

        # Step 4: Load VAEs to CPU
        logger.info("Loading video VAE...")
        self._video_vae_sd = load_vae(video_vae_path, device="cpu", dtype=dtype)
        self._video_vae = self._build_video_vae(self._video_vae_sd, torch.device("cpu"), dtype)

        logger.info("Loading audio VAE...")
        self._audio_vae_sd = load_vae(audio_vae_path, device="cpu", dtype=dtype)
        self._audio_vae = self._build_audio_vae(self._audio_vae_sd, torch.device("cpu"), dtype)

        # Prompt cache
        self._cached_prompt_text = None
        self._cached_context = None

        logger.info(f"LTX 2.3 pipeline loaded in {time.time() - start:.1f}s")
        _log_gpu_memory("all models loaded")

    def _apply_ffn_chunking(self, chunk_size: int):
        for block in self._transformer.transformer_blocks:
            if hasattr(block, 'ff'):
                original_ff = block.ff
                block.ff = _ChunkedFFN(original_ff, chunk_size)
            if hasattr(block, 'audio_ff'):
                original_aff = block.audio_ff
                block.audio_ff = _ChunkedFFN(original_aff, chunk_size)
        logger.info(f"FFN chunking applied with chunk_size={chunk_size}")

    def _move_transformer_to_gpu(self):
        if self._transformer_on_gpu:
            return
        logger.info("Moving transformer to GPU (preserving dtypes)...")
        for param in self._transformer.parameters():
            param.data = param.data.to(device=self.device)
        for buf in self._transformer.buffers():
            buf.data = buf.data.to(device=self.device)
        if hasattr(self._transformer, '_fp8_scales'):
            self._transformer._fp8_scales = {
                k: v.to(device=self.device) for k, v in self._transformer._fp8_scales.items()
            }
        self._transformer_on_gpu = True
        torch.cuda.synchronize()
        _log_gpu_memory("transformer on GPU")

    def _move_transformer_to_cpu(self):
        if not self._transformer_on_gpu:
            return
        logger.info("Moving transformer to CPU...")
        for param in self._transformer.parameters():
            param.data = param.data.to(device="cpu")
        for buf in self._transformer.buffers():
            buf.data = buf.data.to(device="cpu")
        if hasattr(self._transformer, '_fp8_scales'):
            self._transformer._fp8_scales = {
                k: v.to(device="cpu") for k, v in self._transformer._fp8_scales.items()
            }
        self._transformer_on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()
        _log_gpu_memory("transformer offloaded to CPU")

    def _build_video_vae(self, state_dict: dict, device: torch.device, dtype: torch.dtype):
        try:
            import sys
            from pathlib import Path
            modules_dir = Path(__file__).parent / "modules"
            if str(modules_dir) not in sys.path:
                sys.path.insert(0, str(modules_dir))

            from ltx_core.model.video_vae import VideoVAE
            vae = VideoVAE()
            vae.load_state_dict(state_dict, strict=False)
            vae = vae.to(device=device, dtype=dtype)
            vae.eval()
            logger.info("Video VAE loaded via ltx_core")
            return vae
        except ImportError:
            logger.warning("ltx_core not available, storing raw video VAE state dict")
            return _StateDictVAE(state_dict, device, dtype)

    def _build_audio_vae(self, state_dict: dict, device: torch.device, dtype: torch.dtype):
        try:
            import sys
            from pathlib import Path
            modules_dir = Path(__file__).parent / "modules"
            if str(modules_dir) not in sys.path:
                sys.path.insert(0, str(modules_dir))

            from ltx_core.model.audio_vae import AudioVAE
            vae = AudioVAE()
            vae.load_state_dict(state_dict, strict=False)
            vae = vae.to(device=device, dtype=dtype)
            vae.eval()
            logger.info("Audio VAE loaded via ltx_core")
            return vae
        except ImportError:
            logger.warning("ltx_core not available, storing raw audio VAE state dict")
            return _StateDictVAE(state_dict, device, dtype)

    def __call__(self, **kwargs) -> dict:
        return self._generate(**kwargs)

    @torch.inference_mode()
    def _generate(self, **kwargs) -> dict:
        from .text_encoder import encode_prompt

        prompts = kwargs.get("prompts", [{"text": "a beautiful sunset", "weight": 1.0}])
        seed = kwargs.get("seed", kwargs.get("base_seed", 42))
        num_frames_raw = kwargs.get("num_frames", self.num_frames)
        num_frames = round((num_frames_raw - 1) / 8) * 8 + 1
        num_frames = max(num_frames, 9)
        if num_frames != num_frames_raw:
            logger.info(f"Snapped num_frames {num_frames_raw} -> {num_frames} (must be 8K+1)")
        frame_rate = kwargs.get("frame_rate", self.frame_rate)
        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        randomize_seed = kwargs.get("randomize_seed", self.randomize_seed)

        if randomize_seed:
            seed = random.randint(0, 2**31 - 1)
            logger.info(f"Randomized seed: {seed}")

        prompt_text = prompts[0]["text"] if prompts else "a beautiful sunset"

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # =================================================================
        # Text encoding (Gemma on GPU → aggregate embeds → offload)
        # =================================================================
        if self._cached_prompt_text == prompt_text and self._cached_context is not None:
            context = self._cached_context
            logger.debug("Reusing cached prompt encoding")
        else:
            logger.info(f"Encoding prompt: {prompt_text}")

            # Ensure text encoder is on GPU
            if not self._text_encoder_on_gpu:
                logger.info("Loading text encoder back to GPU...")
                self._text_encoder.to(self.device)
                self._text_encoder_on_gpu = True
                torch.cuda.synchronize()

            all_layer_hiddens, attention_mask = encode_prompt(
                self._text_encoder,
                self._tokenizer,
                prompt_text,
                device=self.device,
                dtype=self.dtype,
            )

            # Offload text encoder before running projection
            if self.offload_text_encoder:
                logger.info("Offloading text encoder to CPU...")
                self._text_encoder.to("cpu")
                self._text_encoder_on_gpu = False
                gc.collect()
                torch.cuda.empty_cache()

            # Run text projection (aggregate embeds) on GPU
            self._text_projection.to(self.device)
            self._text_projection_on_gpu = True
            projected = self._text_projection(all_layer_hiddens, attention_mask)
            del all_layer_hiddens

            # Offload text projection
            self._text_projection.to("cpu")
            self._text_projection_on_gpu = False

            # Run embedding connectors on GPU (need transformer on GPU for this)
            self._move_transformer_to_gpu()
            context = self._transformer.preprocess_text_embeds(projected, unprocessed=True)
            # Move transformer back to CPU to free VRAM for latent prep
            self._move_transformer_to_cpu()

            del projected
            gc.collect()
            torch.cuda.empty_cache()

            self._cached_prompt_text = prompt_text
            self._cached_context = context

        # =================================================================
        # Prepare latent noise
        # =================================================================
        vae_temporal_factor = 8
        vae_spatial_factor = 32

        latent_frames = (num_frames - 1) // vae_temporal_factor + 1
        latent_height = height // vae_spatial_factor
        latent_width = width // vae_spatial_factor
        latent_channels = 128

        video_latents = torch.randn(
            (1, latent_channels, latent_frames, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        audio_channels = 8
        audio_freq_bins = 16
        video_duration_sec = num_frames / frame_rate
        audio_latent_time = round(video_duration_sec * 25)
        audio_latents = torch.randn(
            (1, audio_channels, audio_latent_time, audio_freq_bins),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # =================================================================
        # 8-step Euler denoising (transformer on GPU)
        # =================================================================
        sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=self.device, dtype=self.dtype)

        logger.info(f"Denoising {num_frames} frames at {height}x{width} ({len(sigmas)-1} steps)")

        self._move_transformer_to_gpu()

        video_latents, audio_latents = self._euler_denoise(
            video_latents=video_latents,
            audio_latents=audio_latents,
            context=context,
            sigmas=sigmas,
            frame_rate=frame_rate,
        )

        self._move_transformer_to_cpu()

        # =================================================================
        # VAE decode
        # =================================================================
        logger.info("Decoding video from latents...")
        video_tensor = self._decode_video(video_latents)

        logger.info("Decoding audio from latents...")
        audio_tensor = self._decode_audio(audio_latents)

        logger.info(
            f"Generated: video={video_tensor.shape}, audio={audio_tensor.shape}, "
            f"duration={audio_tensor.shape[-1] / AUDIO_SAMPLE_RATE:.2f}s"
        )

        return {
            "video": video_tensor,
            "audio": audio_tensor,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "frame_rate": frame_rate,
        }

    def _euler_denoise(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        sigmas: torch.Tensor,
        frame_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run 8-step Euler sampling with the distilled sigma schedule."""
        v_noisy = video_latents * sigmas[0]
        a_noisy = audio_latents * sigmas[0]

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            timestep = sigma.expand(v_noisy.shape[0])
            a_timestep = sigma.expand(a_noisy.shape[0])

            x_input = [v_noisy, a_noisy]

            model_output = self._transformer(
                x=x_input,
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

            logger.debug(f"Step {i+1}/{len(sigmas)-1}: sigma={sigma:.4f} -> {sigma_next:.4f}")

        return v_noisy, a_noisy

    def _decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        try:
            import sys
            from pathlib import Path
            modules_dir = Path(__file__).parent / "modules"
            if str(modules_dir) not in sys.path:
                sys.path.insert(0, str(modules_dir))

            from ltx_core.model.video_vae import TilingConfig, decode_video as vae_decode_video

            self._video_vae.to(self.device)
            tiling_config = TilingConfig.default()
            decoded = vae_decode_video(latents, self._video_vae, tiling_config)

            frames = []
            for chunk in decoded:
                frames.append(chunk.to(torch.float32))
            video = torch.cat(frames, dim=0)
            video = torch.clamp(video / 255.0, 0.0, 1.0)
            self._video_vae.to("cpu")
            return video

        except ImportError:
            logger.warning("ltx_core not available for video decode, returning raw latents")
            return latents.squeeze(0).permute(1, 2, 3, 0).float()

    def _decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        try:
            import sys
            from pathlib import Path
            modules_dir = Path(__file__).parent / "modules"
            if str(modules_dir) not in sys.path:
                sys.path.insert(0, str(modules_dir))

            from ltx_core.model.audio_vae import decode_audio as vae_decode_audio

            self._audio_vae.to(self.device)
            audio = vae_decode_audio(latents, self._audio_vae, None)
            self._audio_vae.to("cpu")
            return audio

        except ImportError:
            logger.warning("ltx_core not available for audio decode, returning raw latents")
            return latents.squeeze(0).mean(dim=-1).float()


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
            chunk = x[:, i:i + self.chunk_size, :]
            chunks.append(self.ff(chunk))
        return torch.cat(chunks, dim=1)


class _StateDictVAE:
    def __init__(self, state_dict: dict, device: torch.device, dtype: torch.dtype):
        self.state_dict = {k: v.to(device=device, dtype=dtype) for k, v in state_dict.items()}
        self.device = device
        self.dtype = dtype
