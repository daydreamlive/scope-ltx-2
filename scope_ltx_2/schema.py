"""LTX 2.3 pipeline configuration schema."""

from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class LTX2Config(BasePipelineConfig):
    """Configuration for LTX 2.3 audio-video generation pipeline.

    Uses Kijai's separated ComfyUI-format checkpoints (distilled v3 FP8).
    FP8 quantization is baked into the transformer checkpoint, so no
    quantization selection is needed.

    Fits on a 24GB GPU with text encoder offloading enabled.
    """

    pipeline_id: ClassVar[str] = "ltx2"
    pipeline_name: ClassVar[str] = "LTX 2.3"
    pipeline_description: ClassVar[str] = (
        "High-quality audio-video generation with LTX 2.3 (22B distilled)"
    )
    pipeline_version: ClassVar[str] = "0.3.0"
    docs_url: ClassVar[str | None] = "https://github.com/Lightricks/LTX-2"
    estimated_vram_gb: ClassVar[float | None] = 22.0
    requires_models: ClassVar[bool] = True

    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="Kijai/LTX2.3_comfy",
            files=[
                "diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors",
                "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
                "vae/LTX23_video_vae_bf16.safetensors",
                "vae/LTX23_audio_vae_bf16.safetensors",
            ],
        ),
        HuggingfaceRepoArtifact(
            repo_id="google/gemma-3-12b-it",
            files=[
                "config.json",
                "generation_config.json",
                "model-00001-of-00005.safetensors",
                "model-00002-of-00005.safetensors",
                "model-00003-of-00005.safetensors",
                "model-00004-of-00005.safetensors",
                "model-00005-of-00005.safetensors",
                "model.safetensors.index.json",
                "processor_config.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ],
        ),
    ]

    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False
    supports_cache_management: ClassVar[bool] = False
    supports_kv_cache_bias: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = False
    min_dimension: ClassVar[int] = 64
    modified: ClassVar[bool] = False

    modes: ClassVar[dict[str, ModeDefaults]] = {"text": ModeDefaults(default=True)}
    supports_prompts: ClassVar[bool] = True

    # Resolution
    height: int = Field(
        default=512,
        ge=1,
        description="Output height in pixels",
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=768,
        ge=1,
        description="Output width in pixels",
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )

    # LTX-2 VAE requires frame counts following 8K+1 (1, 9, 17, 25, 33, 41, ...)
    num_frames: int = Field(
        default=33,
        ge=9,
        le=257,
        description=(
            "Number of frames to generate per inference call. "
            "LTX-2 works best with 8K+1 values (9, 17, 25, 33, 41, 49, ...). "
            "Other values are snapped to the nearest valid count."
        ),
        json_schema_extra=ui_field_config(
            order=5, label="Frame Count", is_load_param=False
        ),
    )

    frame_rate: float = 24.0

    # FFN chunking for memory-efficient inference
    ffn_chunk_size: int | None = Field(
        default=4096,
        description=(
            "Chunk size for FFN processing. Smaller values use less memory but "
            "have more kernel launch overhead. Set to None to disable chunking."
        ),
    )

    # Text encoder offloading
    offload_text_encoder: bool = Field(
        default=True,
        description=(
            "Offload text encoder to CPU after encoding prompts. "
            "Saves ~25GB VRAM but adds latency when prompts change."
        ),
    )

    # Seed randomization
    randomize_seed: bool = Field(
        default=False,
        description="Randomize seed on every inference call for varied outputs between chunks",
        json_schema_extra=ui_field_config(
            order=9, label="Randomize Seed", is_load_param=False
        ),
    )
