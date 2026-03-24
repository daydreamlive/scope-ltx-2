"""LTX 2.3 pipeline configuration schema."""

from typing import ClassVar

from pydantic import Field, field_validator

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)

VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8


def snap_to_multiple(value: int, factor: int) -> int:
    """Round to the nearest multiple of *factor* (minimum one unit)."""
    return max(factor, round(value / factor) * factor)


def snap_frame_count(value: int) -> int:
    """Snap to the nearest valid LTX-2.3 frame count (N*8+1, minimum 9)."""
    n = max(round((value - 1) / VAE_TEMPORAL_FACTOR), 1)
    return n * VAE_TEMPORAL_FACTOR + 1


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
            repo_id="Comfy-Org/ltx-2",
            files=[
                "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
            ],
        ),
        HuggingfaceRepoArtifact(
            repo_id="google/gemma-3-12b-it",
            files=[
                "config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        ),
    ]

    produces_audio: ClassVar[bool] = True
    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False
    supports_cache_management: ClassVar[bool] = False
    supports_kv_cache_bias: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = False
    min_dimension: ClassVar[int] = 64
    modified: ClassVar[bool] = False

    modes: ClassVar[dict[str, ModeDefaults]] = {"text": ModeDefaults(default=True)}
    supports_prompts: ClassVar[bool] = True

    height: int = Field(
        default=512,
        ge=VAE_SPATIAL_FACTOR,
        description=(
            "Output height in pixels. "
            f"Must be a multiple of {VAE_SPATIAL_FACTOR}; other values are snapped."
        ),
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=768,
        ge=VAE_SPATIAL_FACTOR,
        description=(
            "Output width in pixels. "
            f"Must be a multiple of {VAE_SPATIAL_FACTOR}; other values are snapped."
        ),
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )

    num_frames: int = Field(
        default=33,
        ge=9,
        le=257,
        description=(
            "Number of frames to generate per inference call. "
            "Must follow N*8+1 (9, 17, 25, 33, 41, 49, ...). "
            "Other values are snapped to the nearest valid count."
        ),
        json_schema_extra=ui_field_config(
            order=5, label="Frame Count", is_load_param=False
        ),
    )

    @field_validator("height", "width", mode="before")
    @classmethod
    def _snap_resolution(cls, v: int) -> int:
        return snap_to_multiple(int(v), VAE_SPATIAL_FACTOR)

    @field_validator("num_frames", mode="before")
    @classmethod
    def _snap_num_frames(cls, v: int) -> int:
        return snap_frame_count(int(v))

    frame_rate: float = 24.0

    # FFN chunking for memory-efficient inference
    ffn_chunk_size: int | None = Field(
        default=4096,
        description=(
            "Chunk size for FFN processing. Smaller values use less memory but "
            "have more kernel launch overhead. Set to None to disable chunking."
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
