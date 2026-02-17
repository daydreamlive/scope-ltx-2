"""LTX2 text-to-video pipeline implementation.

This plugin provides LTX2 video generation with synchronized audio for Daydream Scope.

Performance Optimizations:
    - FFN chunking: Reduces activation memory ~10x during denoising
    - NVFP4 quantization: ~4x weight reduction on Blackwell GPUs
    - FP8 quantization: ~2x weight reduction on Ada GPUs
    - Text encoder offloading: Frees ~25GB by moving Gemma to CPU after encoding
    - Weight streaming: Streams transformer blocks from CPU to save VRAM
    - Prompt caching: Reuses encoded prompts when text hasn't changed

Audio Support:
    LTX2 natively supports synchronized audio-video generation. This plugin decodes
    both video and audio latents and returns them as {"video": tensor, "audio": tensor,
    "audio_sample_rate": int} from __call__(). Scope's audio system (AudioProcessingTrack,
    PipelineProcessor.audio_output_queue) handles routing audio to WebRTC and NDI outputs.
"""

import logging
import time
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_model_file_path
from scope.core.pipelines.interface import Pipeline

from .schema import LTX2Config

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class LTX2Pipeline(Pipeline):
    """LTX2 text-to-video generation pipeline.

    This pipeline wraps the LTX2 distilled model for high-quality video generation
    from text prompts. Since LTX2 is not autoregressive, it generates complete videos
    in one shot rather than frame-by-frame.

    Memory Optimization:
    --------------------
    This implementation uses several techniques to reduce memory usage:

    1. **Quantization Options for Weights**:
       - **FP8** (default): ~2x reduction, transformer weights ~25GB (requires Ada SM >= 8.9)
       - **NVFP4**: ~4x reduction, transformer weights ~12GB (requires Blackwell SM >= 10.0)
       - **None**: Full precision BF16, transformer weights ~45GB

    2. **FFN Chunking** (~10x activation memory reduction):
       FFN layers expand hidden dimensions by 4x, creating massive intermediate tensors.
       By processing the sequence in chunks, we reduce peak activation memory from
       ~50GB to ~5GB. This is mathematically identical to standard processing.
       Configure via `ffn_chunk_size` (default: 4096, set to None to disable).

    3. **Text Encoder Offloading**:
       The Gemma 12B text encoder uses ~25GB VRAM but is only needed for encoding.
       After encoding the prompt, the text encoder is offloaded to CPU, freeing
       VRAM for the transformer. Cached prompts avoid reloading entirely.

    4. **Weight Streaming**:
       Transformer blocks can be streamed from CPU to GPU during the forward pass,
       keeping only a subset on GPU at any time. This trades inference speed for
       lower VRAM usage. Configure via `blocks_to_stream`.

    5. **VAE Tiling**: Uses TilingConfig for decoder to reduce peak memory during
       video decoding.

    Memory Breakdown (with all optimizations enabled, FP8):
    -------------------------------------------------------
    - Text encoder: OFFLOADED to CPU (0GB during inference)
    - Transformer weights (cached): ~25GB (FP8) / ~12GB (NVFP4)
    - Video decoder (cached): ~3GB
    - Activations during denoising (with FFN chunking): ~5GB
    - Total: ~33GB (FP8) / ~20GB (NVFP4) -- fits on most GPUs!

    Reference:
    ----------
    - Official LTX-2 documentation:
      https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md
    - FFN chunking technique from ComfyUI:
      https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management
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
        use_fp8: bool = True,
        randomize_seed: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        checkpoint_path: str | None = None,
        gemma_root: str | None = None,
        # New optimization parameters
        quantization=None,
        ffn_chunk_size: int | None = 4096,
        offload_text_encoder: bool = True,
        blocks_to_stream: int = 0,
        prefetch_blocks: int = 1,
        low_vram_init: bool = False,
        **kwargs,  # Accept and ignore unknown params (loras, vae_type, etc.)
    ):
        """Initialize LTX2 pipeline.

        Args:
            height: Output video height in pixels
            width: Output video width in pixels
            num_frames: Number of frames to generate
            frame_rate: Output frame rate
            use_fp8: Enable FP8 quantization for transformer (deprecated, use quantization)
            randomize_seed: Randomize seed on every generation
            device: Target device for inference
            dtype: Data type for model weights
            checkpoint_path: Path to LTX2 checkpoint (auto-detected if None)
            gemma_root: Path to Gemma text encoder (auto-detected if None)
            quantization: Quantization method ("fp8", "nvfp4", or None)
            ffn_chunk_size: Chunk size for FFN processing (None to disable)
            offload_text_encoder: Offload text encoder to CPU after encoding
            blocks_to_stream: Number of transformer blocks to stream from CPU
            prefetch_blocks: Number of blocks to prefetch during streaming
            low_vram_init: Force low-VRAM initialization mode
            **kwargs: Additional parameters (ignored for compatibility)
        """
        import gc
        import sys
        from pathlib import Path

        # Add the modules directory to sys.path so ltx_core and ltx_pipelines can be imported
        modules_dir = Path(__file__).parent / "modules"
        if str(modules_dir) not in sys.path:
            sys.path.insert(0, str(modules_dir))

        from ltx_core.model.video_vae import TilingConfig
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.types import PipelineComponents

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype

        # Store config values as instance attributes
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.randomize_seed = randomize_seed
        self.offload_text_encoder = offload_text_encoder

        # Log ignored kwargs for debugging
        if kwargs:
            ignored = list(kwargs.keys())
            logger.debug(f"LTX2Pipeline ignoring unknown kwargs: {ignored}")

        # Get model paths
        if checkpoint_path is None:
            ltx2_dir = get_model_file_path("LTX-2")
            checkpoint_path = str(ltx2_dir / "ltx-2-19b-distilled.safetensors")

        if gemma_root is None:
            gemma_root = str(get_model_file_path("gemma-3-12b-it"))

        # Initialize model ledger for loading LTX2 components
        start = time.time()
        logger.info(f"Loading LTX2 checkpoint from: {checkpoint_path}")
        logger.info(f"Loading Gemma text encoder from: {gemma_root}")

        # Resolve quantization: enum -> string for ModelLedger
        # Handles Quantization enum, string values, and legacy use_fp8
        from scope.core.pipelines.enums import Quantization
        quantization_value = None
        if quantization is not None:
            if isinstance(quantization, Quantization):
                # Map enum values to ModelLedger's string identifiers
                enum_map = {"fp8_e4m3fn": "fp8", "nvfp4": "nvfp4"}
                quantization_value = enum_map.get(quantization.value)
            elif isinstance(quantization, str):
                if quantization in ("fp8_e4m3fn", "fp8"):
                    quantization_value = "fp8"
                elif quantization == "nvfp4":
                    quantization_value = "nvfp4"

        # Legacy backwards compatibility
        if quantization_value is None and use_fp8:
            quantization_value = "fp8"

        # Store for logging
        self._quantization = quantization_value

        logger.info(
            f"Creating ModelLedger with quantization={quantization_value}, "
            f"ffn_chunk_size={ffn_chunk_size}, low_vram_init={low_vram_init}"
        )

        try:
            self.model_ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
                spatial_upsampler_path=None,
                loras=[],
                quantization=quantization_value,
                ffn_chunk_size=ffn_chunk_size,
                low_vram_init=low_vram_init,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ModelLedger: {e}")
            logger.error(f"Make sure model checkpoint is at: {checkpoint_path}")
            logger.error(f"Make sure Gemma text encoder is at: {gemma_root}")
            raise

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

        # Set up tiling config for VAE decoding
        self.tiling_config = TilingConfig.default()

        # Helper for GPU memory logging
        def log_gpu_memory(stage: str) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"  GPU memory after {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )

        # =====================================================================
        # Load models with memory-efficient ordering
        # =====================================================================
        logger.info("Loading and caching models in VRAM...")

        # Step 1: Load text encoder
        logger.info("  - Loading text encoder (~25GB)...")
        self._cached_text_encoder = self.model_ledger.text_encoder()
        log_gpu_memory("text encoder")

        # Step 2: Offload text encoder to CPU BEFORE loading transformer
        # This prevents OOM when both are in VRAM simultaneously
        self._text_encoder_offloaded = False
        if offload_text_encoder:
            logger.info("  - Offloading text encoder to CPU before transformer load...")
            self._cached_text_encoder.to("cpu")
            self._text_encoder_offloaded = True
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory("text encoder offloaded")

        # Step 3: Load transformer (with quantization + FFN chunking applied)
        logger.info("  - Loading transformer...")
        self._cached_transformer = self.model_ledger.transformer()
        log_gpu_memory("transformer")

        # Force garbage collection after transformer to free temporary tensors
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("GC after transformer")

        # Step 4: Set up weight streaming if configured
        self._streaming_state = None
        if blocks_to_stream > 0:
            from .weight_streaming import BlockStreamingConfig, setup_block_streaming

            logger.info(
                f"Setting up weight streaming: {blocks_to_stream} blocks to stream, "
                f"{prefetch_blocks} prefetch blocks"
            )

            # Get the transformer blocks from the model
            # Structure: X0Model -> velocity_model (LTXModel) -> transformer_blocks
            transformer_blocks = (
                self._cached_transformer.velocity_model.transformer_blocks
            )

            streaming_config = BlockStreamingConfig(
                blocks_to_stream=blocks_to_stream,
                prefetch_blocks=prefetch_blocks,
                use_pinned_memory=True,
                use_non_blocking=True,
                compute_device=self.device,
                debug=False,
            )

            self._streaming_state = setup_block_streaming(
                transformer_blocks, streaming_config
            )
            log_gpu_memory("weight streaming setup")

        # Step 5: Load decoders
        logger.info("  - Loading video decoder (~3GB)...")
        self._cached_video_decoder = self.model_ledger.video_decoder()


        log_gpu_memory("all models")
        logger.info("All models cached successfully")
        # Load audio decoder and vocoder for synchronized A/V output
        logger.info("  - Loading audio decoder (~1GB)...")
        self._cached_audio_decoder = self.model_ledger.audio_decoder()
        logger.info("  - Loading vocoder (~0.5GB)...")
        self._cached_vocoder = self.model_ledger.vocoder()
        logger.info("All models cached successfully in VRAM")

        logger.info(f"LTX2 models loaded in {time.time() - start:.2f}s")

        # Log quantization status
        if self._quantization == "nvfp4":
            logger.info(
                "NVFP4 quantization: enabled (Blackwell GPU SM >= 10.0, comfy-kitchen)"
            )
        elif self._quantization == "fp8":
            logger.info("FP8 quantization: enabled")
        else:
            logger.info("Quantization: disabled (full precision BF16)")

        if ffn_chunk_size is not None:
            logger.info(f"FFN chunking: enabled (chunk_size={ffn_chunk_size})")
        else:
            logger.info("FFN chunking: disabled")

        if offload_text_encoder:
            logger.info("Text encoder offloading: enabled (saves ~25GB during inference)")

        if blocks_to_stream > 0:
            logger.info(f"Weight streaming: {blocks_to_stream} blocks streamed from CPU")

    def __call__(self, **kwargs) -> dict:
        """Generate synchronized video and audio from text prompt.

        Args:
            **kwargs: Generation parameters including:
                - prompts: List of prompt dictionaries with 'text' and 'weight' keys
                - seed: Random seed for generation
                - num_frames: Number of frames to generate (overrides config)
                - frame_rate: Frame rate for video (overrides config)

        Returns:
            Dictionary with:
                - "video": [T, H, W, C] tensor in [0, 1] range
                - "audio": [C, S] tensor in [-1, 1] range (if audio was generated)
                - "audio_sample_rate": int, native sample rate of the audio (e.g. 24000)
        """
        return self._generate(**kwargs)

    @torch.inference_mode()
    def _generate(self, **kwargs) -> dict:
        """Internal generation method."""
        import gc
        import random

        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import (
            AUDIO_SAMPLE_RATE,
            DISTILLED_SIGMA_VALUES,
        )
        from ltx_pipelines.utils.helpers import (
            denoise_audio_video,
            euler_denoising_loop,
            simple_denoising_func,
        )

        # Extract parameters
        prompts = kwargs.get("prompts", [{"text": "a beautiful sunset", "weight": 1.0}])
        seed = kwargs.get("seed", kwargs.get("base_seed", 42))
        num_frames = kwargs.get("num_frames", self.num_frames)
        frame_rate = kwargs.get("frame_rate", self.frame_rate)
        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        randomize_seed = kwargs.get("randomize_seed", self.randomize_seed)

        # Randomize seed if enabled (useful for non-autoregressive models like LTX2)
        # This ensures each chunk gets a different seed for varied outputs
        if randomize_seed:
            seed = random.randint(0, 2**31 - 1)
            logger.info(f"Randomized seed: {seed}")

        # Convert prompts to single text (for now, just use first prompt)
        prompt_text = prompts[0]["text"] if prompts else "a beautiful sunset"

        # Set up generator and components
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # =====================================================================
        # Prompt encoding with caching + text encoder offloading
        # =====================================================================
        cached_prompt = getattr(self, "_cached_prompt_text", None)

        if cached_prompt == prompt_text and hasattr(self, "_cached_context"):
            # Reuse cached encoding - no need to load text encoder
            video_context, audio_context = self._cached_context
            logger.debug("Reusing cached prompt encoding")
        else:
            # Prompt changed - need to re-encode
            logger.info(f"Encoding prompt: {prompt_text}")

            # Check if text encoder needs to be loaded back to GPU
            if self.offload_text_encoder and self._text_encoder_offloaded:
                logger.info("Loading text encoder back to GPU for encoding...")
                self._cached_text_encoder.to(self.device)
                self._text_encoder_offloaded = False
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            context_p = encode_text(self._cached_text_encoder, prompts=[prompt_text])[0]
            video_context, audio_context = context_p

            # Cache the encoded prompt
            self._cached_prompt_text = prompt_text
            self._cached_context = (video_context, audio_context)

            # Offload text encoder to CPU to free VRAM for transformer
            if self.offload_text_encoder:
                logger.info("Offloading text encoder to CPU to free VRAM...")
                self._cached_text_encoder.to("cpu")
                self._text_encoder_offloaded = True
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"GPU memory after text encoder offload: {allocated:.2f}GB"
                    )

        # Use cached transformer for generation
        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        # Define denoising loop
        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=self._cached_transformer,
                ),
            )

        # Set up output shape (LTX2 generates at full resolution in one stage for simplicity)
        output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )

        # No image conditioning for now
        conditionings = []

        # Generate video and audio latents
        logger.info(f"Generating {num_frames} frames at {height}x{width}")
        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
        )

        # Decode video from latents using cached decoder
        logger.info("Decoding video from latents")

        # Use tiling for VAE decoding to reduce memory usage
        decoded_video = vae_decode_video(
            video_state.latent, self._cached_video_decoder, self.tiling_config
        )

        # Convert decoded video iterator to tensor and postprocess
        # LTX2 vae_decode_video returns an iterator of frame chunks
        video_frames = []
        for chunk in decoded_video:
            video_frames.append(chunk.to(torch.float32))

        # Concatenate all chunks along time dimension -> [T, H, W, C]
        video_tensor = torch.cat(video_frames, dim=0)

        # Normalize from [0, 255] uint8 to [0, 1] float
        video_tensor = torch.clamp(video_tensor / 255.0, 0.0, 1.0)

        # Decode audio from latents using cached audio decoder and vocoder
        # Following the official LTX-2 pipeline pattern — always decode audio
        logger.info("Decoding audio from latents")
        audio_tensor = vae_decode_audio(
            audio_state.latent,
            self._cached_audio_decoder,
            self._cached_vocoder,
        )
        # audio_tensor shape: (channels, samples) - typically (2, N) for stereo at 24kHz
        logger.info(
            f"Audio decoded: shape={audio_tensor.shape}, "
            f"sample_rate={AUDIO_SAMPLE_RATE}, "
            f"duration={audio_tensor.shape[-1] / AUDIO_SAMPLE_RATE:.3f}s"
        )

        return {
            "video": video_tensor,
            "audio": audio_tensor,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
        }
