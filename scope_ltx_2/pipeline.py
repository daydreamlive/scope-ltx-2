"""LTX2 text-to-video pipeline implementation.

This plugin provides LTX2 video generation with synchronized audio for Daydream Scope.

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
    This implementation is optimized for maximum inference speed by keeping all models
    in VRAM:

    1. **FP8 Quantization for Weights Only**: Enabled by default (use_fp8=True) to
       reduce transformer weights from ~45GB to ~25GB. However, **activations during
       inference remain in BF16** and are the main memory bottleneck.

    2. **All Models Cached in VRAM**: Text encoder (~20GB), transformer (~25GB), and
       video decoder (~3GB) are loaded once during initialization and kept in VRAM.
       This eliminates all model loading overhead between generations, providing
       maximum inference speed at the cost of higher baseline VRAM usage.

    3. **PYTORCH_CUDA_ALLOC_CONF**: Set to "expandable_segments:True" in app.py to
       prevent memory fragmentation with FP8 quantization.

    4. **VAE Tiling**: Uses TilingConfig for decoder to reduce peak memory during
       video decoding.

    5. **No Unnecessary Models**: Video encoder is only loaded for i2v conditioning.

    6. **Minimal Defaults**: 33 frames at 512x768 to fit in 96GB VRAM.
       **Activations are the bottleneck**: ~1.5GB per frame at 512x768.

    CRITICAL LIMITATION:
    --------------------
    Unlike other pipelines that use torchao FP8 quantization for both weights AND
    activations, LTX2's custom FP8 only quantizes weights. This means the transformer's
    intermediate activations during denoising consume 60-80GB at higher resolutions.

    Memory Breakdown (96GB GPU):
    ----------------------------
    - Text encoder (cached): ~20GB
    - Transformer weights FP8 (cached): ~25GB
    - Video decoder (cached): ~3GB
    - **Activations during denoising**:
      * 33 frames @ 512x768: ~50GB ✅ Fits (total ~98GB)
      * 61 frames @ 768x1024: ~90GB ❌ OOM (total ~138GB)
      * 121 frames @ 1024x1536: ~150GB ❌ OOM (total ~198GB)

    Baseline VRAM = Text Encoder (20GB) + Transformer (25GB) + Decoder (3GB) = 48GB
    Peak VRAM = Baseline (48GB) + Activations (resolution × frames dependent)

    For higher quality, you need:
    - A GPU with 141GB+ VRAM (H100)
    - OR the two-stage pipeline (low-res generation + upsampling)
    - OR generate shorter/lower-res videos

    Reference:
    ----------
    Official LTX-2 documentation:
    https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md
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
        **kwargs,  # Accept and ignore unknown params (loras, vae_type, etc.)
    ):
        """Initialize LTX2 pipeline.

        Args:
            height: Output video height in pixels
            width: Output video width in pixels
            num_frames: Number of frames to generate
            frame_rate: Output frame rate
            use_fp8: Enable FP8 quantization for transformer
            randomize_seed: Randomize seed on every generation
            device: Target device for inference
            dtype: Data type for model weights
            checkpoint_path: Path to LTX2 checkpoint (auto-detected if None)
            gemma_root: Path to Gemma text encoder (auto-detected if None)
            **kwargs: Additional parameters (ignored for compatibility)
        """
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
        self.use_fp8 = use_fp8
        self.randomize_seed = randomize_seed

        # Log ignored kwargs for debugging
        if kwargs:
            ignored = list(kwargs.keys())
            logger.debug(f"LTX2Pipeline ignoring unknown kwargs: {ignored}")

        # Get model paths
        # Models are downloaded to:
        # - LTX-2/ltx-2-19b-distilled.safetensors (main model checkpoint)
        # - gemma-3-12b-it/ (contains tokenizer and model files)
        if checkpoint_path is None:
            # Use scope's config helper to get model paths
            ltx2_dir = get_model_file_path("LTX-2")
            checkpoint_path = str(ltx2_dir / "ltx-2-19b-distilled.safetensors")

        if gemma_root is None:
            gemma_root = str(get_model_file_path("gemma-3-12b-it"))

        # Initialize model ledger for loading LTX2 components
        start = time.time()
        logger.info(f"Loading LTX2 checkpoint from: {checkpoint_path}")
        logger.info(f"Loading Gemma text encoder from: {gemma_root}")

        # Enable FP8 quantization to reduce VRAM usage
        # According to official LTX-2 docs, this significantly reduces memory footprint
        fp8_enabled = self.use_fp8

        try:
            self.model_ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
                spatial_upsampler_path=None,  # We'll add upsampler support later
                loras=[],
                # Use default DummyRegistry - don't cache state dicts in RAM
                fp8transformer=fp8_enabled,  # FP8 significantly reduces VRAM usage
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

        # Cache all models in VRAM for maximum performance
        # This uses more VRAM (~48GB total) but eliminates all reload overhead
        logger.info("Loading and caching models in VRAM...")
        logger.info("  - Loading text encoder (~20GB)...")
        self._cached_text_encoder = self.model_ledger.text_encoder()
        logger.info("  - Loading transformer (~25GB)...")
        self._cached_transformer = self.model_ledger.transformer()
        logger.info("  - Loading video decoder (~3GB)...")
        self._cached_video_decoder = self.model_ledger.video_decoder()

        # Load audio decoder and vocoder for synchronized A/V output
        logger.info("  - Loading audio decoder (~1GB)...")
        self._cached_audio_decoder = self.model_ledger.audio_decoder()
        logger.info("  - Loading vocoder (~0.5GB)...")
        self._cached_vocoder = self.model_ledger.vocoder()
        logger.info("All models cached successfully in VRAM")

        logger.info(f"LTX2 models loaded in {time.time() - start:.2f}s")
        logger.info(f"FP8 quantization: {'enabled' if fp8_enabled else 'disabled'}")
        if fp8_enabled:
            logger.warning(
                "FP8 quantization only reduces weight size (~25GB). "
                "Activations during inference are still in BF16 and are the main memory bottleneck. "
                f"At {self.height}x{self.width} with {self.num_frames} frames, "
                "expect ~50-60GB for activations during denoising."
            )

        # NOTE: This is currently a single-stage pipeline implementation.
        # For even lower VRAM usage, consider implementing a two-stage pipeline:
        # - Stage 1: Generate at lower resolution (512x768) with CFG guidance
        # - Stage 2: Upsample to full resolution (1024x1536) with distilled LoRA
        # See: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py

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

        # Encode text prompt using cached text encoder
        logger.info(f"Encoding prompt: {prompt_text}")
        context_p = encode_text(self._cached_text_encoder, prompts=[prompt_text])[0]
        video_context, audio_context = context_p

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
