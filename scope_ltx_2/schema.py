"""LTX 2.3 pipeline configuration schema."""

import math
from typing import ClassVar, Literal

from pydantic import Field, field_validator

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)

VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8

DISTILLED_SIGMAS: list[float] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
]
DISTILLED_NUM_STEPS = len(DISTILLED_SIGMAS) - 1  # 8

SigmaSchedule = Literal[
    "distilled", "linear", "cosine", "linear_quadratic", "beta",
]


def _linear_sigmas(n: int) -> list[float]:
    return [1.0 - i / n for i in range(n + 1)]


def _cosine_sigmas(n: int) -> list[float]:
    """Cosine schedule — spends more steps in the mid-noise range."""
    return [math.cos(i / n * math.pi / 2) for i in range(n + 1)]


def _linear_quadratic_sigmas(
    n: int, threshold_noise: float = 0.025, linear_steps: int | None = None,
) -> list[float]:
    """Two-phase schedule: linear ramp then quadratic curve.

    Ported from ComfyUI's ``linear_quadratic_schedule`` (originally from the
    genmo/Mochi codebase).  Operates directly in the [0, 1] sigma range so
    no model_sampling lookup is needed.
    """
    if n == 1:
        return [1.0, 0.0]
    if linear_steps is None:
        linear_steps = n // 2
    linear_part = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_diff = linear_steps - threshold_noise * n
    quad_steps = n - linear_steps
    quad_coef = threshold_diff / (linear_steps * quad_steps ** 2)
    lin_coef = threshold_noise / linear_steps - 2 * threshold_diff / (quad_steps ** 2)
    const = quad_coef * (linear_steps ** 2)
    quad_part = [
        quad_coef * (i ** 2) + lin_coef * i + const
        for i in range(linear_steps, n)
    ]
    ascending = linear_part + quad_part + [1.0]
    return [1.0 - x for x in ascending]


def _beta_sigmas(n: int, alpha: float = 0.6, beta: float = 0.6) -> list[float]:
    """Beta-distribution schedule adapted from ComfyUI's beta_scheduler.

    Approximates ``scipy.stats.beta.ppf`` (the quantile / inverse-CDF
    function) via bisection on a power-series CDF so we avoid a scipy
    dependency.
    """
    def _betainc(a: float, b: float, x: float, terms: int = 200) -> float:
        """Power-series approximation of the regularised incomplete beta I_x(a,b)."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        prefix = math.exp(a * math.log(x) + (b - 1) * math.log(1 - x) - log_beta) / a
        s, numerator = 1.0, 1.0
        for k in range(1, terms):
            numerator *= (k - b) * x / k
            term = numerator / (a + k)
            s += term
            if abs(term) < 1e-12:
                break
        return min(max(prefix * s, 0.0), 1.0)

    def _beta_ppf(a: float, b: float, p: float) -> float:
        """Inverse CDF of Beta(a, b) at probability *p* via bisection."""
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if _betainc(a, b, mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    quantiles = [1.0 - i / n for i in range(n)]
    sigmas = [_beta_ppf(alpha, beta, q) for q in quantiles]
    sigmas.append(0.0)
    return sigmas


def make_sigma_schedule(
    num_steps: int,
    schedule: SigmaSchedule = "distilled",
) -> list[float]:
    """Build a sigma schedule for *num_steps* Euler denoising steps."""
    if schedule == "distilled":
        if num_steps == DISTILLED_NUM_STEPS:
            return list(DISTILLED_SIGMAS)
        return _linear_sigmas(num_steps)
    if schedule == "linear":
        return _linear_sigmas(num_steps)
    if schedule == "cosine":
        return _cosine_sigmas(num_steps)
    if schedule == "linear_quadratic":
        return _linear_quadratic_sigmas(num_steps)
    if schedule == "beta":
        return _beta_sigmas(num_steps)
    return _linear_sigmas(num_steps)


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
    supports_lora: ClassVar[bool] = True
    supports_vace: ClassVar[bool] = False
    supports_cache_management: ClassVar[bool] = False
    supports_kv_cache_bias: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = False
    min_dimension: ClassVar[int] = 64
    modified: ClassVar[bool] = False

    modes: ClassVar[dict[str, ModeDefaults]] = {"text": ModeDefaults(default=True)}
    supports_prompts: ClassVar[bool] = True

    lora_merge_strategy: str = Field(
        default="permanent_merge",
        description="LoRA merge strategy (only permanent_merge is supported for FP8 models)",
        json_schema_extra=ui_field_config(
            order=3, component="lora", is_load_param=True
        ),
    )

    height: int = Field(
        default=384,
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
        default=320,
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
        default=129,
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

    num_steps: int = Field(
        default=DISTILLED_NUM_STEPS,
        ge=1,
        le=20,
        description="Number of Euler denoising steps.",
        json_schema_extra=ui_field_config(
            order=6, label="Denoising Steps", is_load_param=False
        ),
    )

    schedule: SigmaSchedule = Field(
        default="distilled",
        description=(
            "Sigma schedule type. 'distilled' uses the pre-trained 8-step "
            "schedule (falls back to linear for other step counts). "
            "'linear', 'cosine', 'linear_quadratic', and 'beta' generate "
            "schedules for any step count."
        ),
        json_schema_extra=ui_field_config(
            order=7, label="Sigma Schedule", is_load_param=False
        ),
    )

    sigmas: list[float] | None = Field(
        default=None,
        description=(
            "Custom sigma schedule for Euler denoising (API only). "
            "When provided, overrides num_steps and schedule. Must be a "
            "descending list from 1.0 to 0.0; the number of steps is "
            "len(sigmas)-1."
        ),
    )

    @field_validator("sigmas", mode="before")
    @classmethod
    def _validate_sigmas(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) < 2:
            raise ValueError("sigmas must have at least 2 values (1 denoising step)")
        return v

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
        default=True,
        description="Randomize seed on every inference call for varied outputs between chunks",
        json_schema_extra=ui_field_config(
            order=9, label="Randomize Seed", is_load_param=False
        ),
    )
