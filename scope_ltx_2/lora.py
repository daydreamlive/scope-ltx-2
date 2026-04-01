"""LoRA support for LTX 2.3 via direct weight merging.

Permanently merges LoRA deltas into model parameters at load time.
FP8 quantized layers are dequantized, merged, and re-quantized in place
so the patched ``_fp8_scaled_forward`` remains correct.  Compatible with
block streaming and incurs zero runtime overhead.

When multiple LoRAs are combined (e.g. style + IC-LoRA), all deltas are
accumulated per layer in float32 before a single FP8 requantization,
avoiding precision loss from intermediate quantization steps.
"""

import logging
from collections import defaultdict
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


def restore_original_weights(snapshot: dict[str, dict[str, Any]]) -> None:
    """Restore original weights from a snapshot created during LoRA merge.

    Uses stored module references directly, so this works even after
    FFN chunking changes the module hierarchy paths.
    """
    for name, entry in snapshot.items():
        module = entry["module"]
        module.weight.data.copy_(entry["weight"])
        if "fp8_weight_scale" in entry:
            module._fp8_weight_scale = entry["fp8_weight_scale"].clone()
    logger.info("Restored original weights for %d layers", len(snapshot))


def load_and_merge_loras(
    model: nn.Module,
    lora_configs: list[dict[str, Any]],
    linear_modules: dict[str, nn.Linear] | None = None,
    save_snapshot: dict | None = None,
) -> list[dict[str, Any]]:
    """Load LoRA files and permanently merge weights into *model*.

    All LoRA deltas targeting the same layer are accumulated in float32
    before a single FP8 requantization, avoiding precision loss from
    intermediate quantization steps when combining multiple LoRAs.

    Args:
        model: The transformer model to merge into.
        lora_configs: List of LoRA config dicts with ``path`` and ``scale``.
        linear_modules: Pre-built name-to-module mapping.  When provided,
            LoRA keys are matched against these names instead of the
            current ``model.named_modules()`` hierarchy (useful when FFN
            chunking has altered the module paths after init).
        save_snapshot: If provided (empty dict), original weights for
            each modified layer are saved *before* applying deltas,
            enabling later restoration via ``restore_original_weights``
            for a clean single-pass re-merge.

    Returns a list of dicts with ``path``, ``scale``, and
    ``reference_downscale_factor`` per merged LoRA.
    """
    if not lora_configs:
        return []

    if linear_modules is None:
        linear_modules = {
            name: mod
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
        }
    loaded: list[dict[str, Any]] = []

    # Phase 1: collect all deltas per layer across all LoRA files
    layer_deltas: dict[str, list[torch.Tensor]] = defaultdict(list)

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

        matched = 0
        for model_key, info in pairs.items():
            if model_key not in linear_modules:
                continue

            lora_A, lora_B = info["lora_A"].float(), info["lora_B"].float()
            alpha_factor = (info["alpha"] / info["rank"]) if info["alpha"] is not None else 1.0
            delta = (lora_B @ lora_A) * (alpha_factor * scale)
            layer_deltas[model_key].append(delta)
            matched += 1

        logger.info("Collected %d/%d LoRA deltas from %s", matched, len(pairs), Path(path).name)
        loaded.append({
            "path": str(path),
            "scale": scale,
            "reference_downscale_factor": reference_downscale_factor,
        })

    # Phase 2: apply accumulated deltas with single requantization per layer
    merged = 0
    for model_key, deltas in layer_deltas.items():
        module = linear_modules[model_key]

        if save_snapshot is not None and model_key not in save_snapshot:
            entry: dict[str, Any] = {
                "module": module,
                "weight": module.weight.data.clone(),
            }
            if hasattr(module, "_fp8_weight_scale"):
                entry["fp8_weight_scale"] = module._fp8_weight_scale.clone()
            save_snapshot[model_key] = entry

        if module.weight.dtype == torch.float8_e4m3fn:
            ws = getattr(module, "_fp8_weight_scale", torch.tensor(1.0))
            w = module.weight.data.float() * ws.float()
            for delta in deltas:
                w += delta
            amax = w.abs().amax().clamp(min=1e-12)
            new_ws = amax / _FP8_MAX
            module.weight.data = (w / new_ws).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            module._fp8_weight_scale = new_ws
        else:
            w = module.weight.data.float()
            for delta in deltas:
                w += delta
            module.weight.data = w.to(module.weight.dtype)

        merged += 1

    logger.info(
        "Batch-merged %d layers from %d LoRA file(s)",
        merged, len(lora_configs),
    )
    return loaded
