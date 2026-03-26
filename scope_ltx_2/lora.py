"""LoRA support for LTX 2.3 via direct weight merging.

Permanently merges LoRA deltas into model parameters at load time.
FP8 quantized layers are dequantized, merged, and re-quantized in place
so the patched ``_fp8_scaled_forward`` remains correct.  Compatible with
block streaming and incurs zero runtime overhead.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

_STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _extract_lora_pairs(
    sd: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Parse a LoRA state dict into ``{model_key: {lora_A, lora_B, alpha, rank}}``."""
    pairs: dict[str, dict[str, Any]] = {}

    for key in sd:
        # Identify the "down" matrix and derive the matching "up" key
        if ".lora_down.weight" in key:
            base = key.replace(".lora_down.weight", "")
            up_key = f"{base}.lora_up.weight"
        elif ".lora_A.weight" in key:
            base = key.replace(".lora_A.weight", "")
            up_key = f"{base}.lora_B.weight"
        else:
            continue

        if up_key not in sd or base in pairs:
            continue

        # Strip common prefixes to get model-relative path
        model_key = base
        for prefix in _STRIP_PREFIXES:
            if model_key.startswith(prefix):
                model_key = model_key[len(prefix) :]
                break

        alpha_key = f"{base}.alpha"
        pairs[model_key] = {
            "lora_A": sd[key],
            "lora_B": sd[up_key],
            "alpha": sd[alpha_key].item() if alpha_key in sd else None,
            "rank": sd[key].shape[0],
        }

    return pairs


def _extract_safetensors_metadata(path: str) -> dict[str, str]:
    """Read safetensors file-level metadata (not tensor data)."""
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata()) if f.metadata() else {}


def load_and_merge_loras(
    model: nn.Module,
    lora_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load LoRA files and permanently merge weights into *model*.

    For FP8 layers the weight is dequantized via the module's stored
    ``_fp8_weight_scale``, the delta is added, and the result is
    re-quantized.  The per-tensor scale is updated on the module so the
    existing monkey-patched forward (``_fp8_scaled_forward``) stays correct.

    Returns a list of dicts with ``path``, ``scale``, and
    ``reference_downscale_factor`` (extracted from safetensors metadata
    for IC-LoRA checkpoints; defaults to ``1.0``).
    """
    if not lora_configs:
        return []

    linear_modules = {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    }
    loaded: list[dict[str, Any]] = []

    for cfg in lora_configs:
        path = cfg.get("path")
        scale = float(cfg.get("scale", 1.0))

        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"LoRA file not found: {path}. "
                "Ensure the file exists in the models/lora/ directory."
            )

        logger.info("Loading LoRA: %s (scale=%.2f)", Path(path).name, scale)

        metadata = _extract_safetensors_metadata(str(path))
        try:
            reference_downscale_factor = float(metadata["reference_downscale_factor"])
            logger.info(
                "IC-LoRA detected: reference_downscale_factor=%.1f from %s",
                reference_downscale_factor, Path(path).name,
            )
        except (KeyError, ValueError, TypeError):
            reference_downscale_factor = 1.0

        sd = load_file(str(path), device="cpu")
        pairs = _extract_lora_pairs(sd)
        del sd

        merged = 0
        for model_key, info in pairs.items():
            module = linear_modules.get(model_key)
            if module is None:
                continue

            lora_A, lora_B = info["lora_A"].float(), info["lora_B"].float()
            alpha_factor = (info["alpha"] / info["rank"]) if info["alpha"] is not None else 1.0
            delta = (lora_B @ lora_A) * (alpha_factor * scale)

            if module.weight.dtype == torch.float8_e4m3fn:
                ws = getattr(module, "_fp8_weight_scale", torch.tensor(1.0))
                w = module.weight.data.float() * ws.float() + delta
                amax = w.abs().amax().clamp(min=1e-12)
                new_ws = amax / _FP8_MAX
                module.weight.data = (w / new_ws).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
                module._fp8_weight_scale = new_ws
            else:
                module.weight.data = (module.weight.data.float() + delta).to(module.weight.dtype)

            merged += 1

        logger.info("Merged %d/%d LoRA weights from %s", merged, len(pairs), Path(path).name)
        loaded.append({
            "path": str(path),
            "scale": scale,
            "reference_downscale_factor": reference_downscale_factor,
        })

    return loaded
