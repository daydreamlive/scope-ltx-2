"""Model loading utilities for LTX 2.3 with Kijai separated checkpoints.

Handles loading the transformer, VAE, and text projection from separate
safetensors files in the Kijai/LTX2.3_comfy format.
"""

import json
import logging
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .ltx_model import LTXAVModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP8 scaled linear patching
# ---------------------------------------------------------------------------

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _fp8_scaled_forward(
    self: nn.Linear,
    input: torch.Tensor,
) -> torch.Tensor:
    """Forward pass using torch._scaled_mm for FP8 quantized weights.

    The weight is stored in float8_e4m3fn with per-tensor input_scale and
    weight_scale. We quantize the input to FP8 using the stored input_scale,
    then use _scaled_mm for the matmul, and add bias in the output dtype.
    """
    weight_t = self.weight.data.t()
    input_scale: torch.Tensor = self._fp8_input_scale
    weight_scale: torch.Tensor = self._fp8_weight_scale

    orig_shape = input.shape
    orig_dtype = input.dtype
    if input.ndim > 2:
        input = input.reshape(-1, orig_shape[-1])

    dev = input.device
    input_scale = input_scale.to(dev, non_blocking=True)
    weight_scale = weight_scale.to(dev, non_blocking=True)

    input_fp8 = (input.float() / input_scale).clamp(
        -_FP8_MAX, _FP8_MAX,
    ).to(torch.float8_e4m3fn).contiguous()

    out = torch._scaled_mm(
        input_fp8,
        weight_t,
        scale_a=input_scale,
        scale_b=weight_scale,
        out_dtype=orig_dtype,
    )
    if isinstance(out, tuple):
        out = out[0]

    if self.bias is not None:
        out = out + self.bias.to(out.dtype)

    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], out.shape[-1])

    return out


def attach_fp8_scales(model: nn.Module, fp8_scales: dict[str, torch.Tensor]) -> int:
    """Attach FP8 per-tensor scales as module attributes (no forward override).

    LoRA merge (``lora.py``) and the per-block int4 re-quant
    (``quantize_transformer_int4``) both look up the per-tensor weight scale
    via ``module._fp8_weight_scale``.  Keeping that lookup module-local — and
    letting LoRA merge update it in place — means we don't have to thread the
    ``model._fp8_scales`` dict through both paths or worry about it going
    stale after a merge.
    """
    _default_input_scale = torch.tensor(1.0, dtype=torch.float32)
    attached = 0
    skipped = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.dtype != torch.float8_e4m3fn:
            continue

        ws_key = f"{name}.weight_scale"
        if ws_key not in fp8_scales:
            logger.warning(f"FP8 layer {name} missing weight_scale, skipping")
            skipped += 1
            continue

        is_key = f"{name}.input_scale"
        module._fp8_input_scale = fp8_scales.get(is_key, _default_input_scale)
        module._fp8_weight_scale = fp8_scales[ws_key]
        attached += 1

    if skipped:
        logger.warning(f"{skipped} FP8 layers skipped (missing weight_scale)")
    return attached


def patch_fp8_layers(model: nn.Module, fp8_scales: dict[str, torch.Tensor]) -> int:
    """Attach FP8 scales and override forward to use ``torch._scaled_mm``.

    If input_scale is absent (e.g. Comfy-Org Gemma FP8 checkpoints),
    defaults to 1.0 — matching ComfyUI's fp8_linear which clamps input
    to FP8 range and casts directly without prescaling.

    Returns the number of layers patched.
    """
    attach_fp8_scales(model, fp8_scales)
    patched = 0
    for _name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.dtype != torch.float8_e4m3fn:
            continue
        if not hasattr(module, "_fp8_weight_scale"):
            continue
        module.forward = types.MethodType(_fp8_scaled_forward, module)
        patched += 1
    return patched


def detect_config_from_state_dict(state_dict: dict, metadata: dict | None = None) -> dict:
    """Detect LTXAVModel config from checkpoint state dict keys.

    Mirrors ComfyUI's model_detection.py logic for lightricks models.
    Handles both full checkpoints and Kijai separated transformer-only checkpoints.
    """
    keys = set(state_dict.keys())

    is_av = "audio_adaln_single.linear.weight" in keys

    num_layers = 0
    while f"transformer_blocks.{num_layers}.attn2.to_k.weight" in keys:
        num_layers += 1

    attn2_k_shape = state_dict["transformer_blocks.0.attn2.to_k.weight"].shape
    attention_head_dim = attn2_k_shape[0] // 32
    cross_attention_dim = attn2_k_shape[1]

    config = {
        "num_layers": num_layers,
        "attention_head_dim": attention_head_dim,
        "cross_attention_dim": cross_attention_dim,
    }

    if metadata is not None and "config" in metadata:
        try:
            extra = json.loads(metadata["config"]).get("transformer", {})
            config.update(extra)
        except (json.JSONDecodeError, AttributeError):
            pass

    if is_av:
        config.setdefault("in_channels", 128)
        config.setdefault("audio_in_channels", 128)
        config.setdefault("audio_attention_head_dim", 64)
        config.setdefault("num_attention_heads", 32)
        config.setdefault("audio_num_attention_heads", 32)
        config.setdefault("positional_embedding_theta", 10000.0)
        config.setdefault("positional_embedding_max_pos", [20, 2048, 2048])
        config.setdefault("audio_positional_embedding_max_pos", [20])
        config.setdefault("causal_temporal_positioning", False)
        config.setdefault("vae_scale_factors", (8, 32, 32))
        config.setdefault("use_middle_indices_grid", False)
        config.setdefault("timestep_scale_multiplier", 1000.0)
        config.setdefault("av_ca_timestep_scale_multiplier", 1.0)
        config.setdefault("apply_gated_attention", False)

        # Detect audio cross attention dim from audio_attn2.to_k weight
        audio_attn2_key = "transformer_blocks.0.audio_attn2.to_k.weight"
        if audio_attn2_key in keys:
            audio_attn2_shape = state_dict[audio_attn2_key].shape
            config.setdefault("audio_cross_attention_dim", audio_attn2_shape[1])
        else:
            config.setdefault("audio_cross_attention_dim", 2048)

        # Detect caption_channels from caption_projection if present in checkpoint
        if "caption_projection.linear_1.weight" in keys:
            config.setdefault("caption_channels", state_dict["caption_projection.linear_1.weight"].shape[1])
            config.setdefault("caption_proj_before_connector", False)
        elif "caption_projection.linear_1.weight" not in keys:
            # Kijai separated format: text projection is in a separate file.
            # The transformer-only checkpoint won't have caption_projection keys,
            # but the model still needs caption_proj_before_connector=True so
            # it creates NormSingleLinearTextProjection + embedding connectors.
            config.setdefault("caption_proj_before_connector", True)
            config.setdefault("caption_channels", 3840)

        # Detect cross_attention_adaln from scale_shift_table shape
        config.setdefault("cross_attention_adaln", False)
        if "transformer_blocks.0.scale_shift_table" in keys:
            sst_shape = state_dict["transformer_blocks.0.scale_shift_table"].shape
            if sst_shape[0] == 9:
                config["cross_attention_adaln"] = True

        if "transformer_blocks.0.attn1.to_gate_logits.weight" in keys:
            config["apply_gated_attention"] = True

        # LTX 2.3 uses split RoPE and float64 precision for frequency generation
        config.setdefault("rope_type", "split")
        config.setdefault("frequencies_precision", "float64")

    logger.info(
        f"Detected LTX{'AV' if is_av else 'V'} config: "
        f"layers={num_layers}, head_dim={attention_head_dim}, "
        f"cross_dim={cross_attention_dim}, "
        f"caption_proj_before_connector={config.get('caption_proj_before_connector')}, "
        f"adaln={'cross' if config.get('cross_attention_adaln') else 'base'}"
    )

    return config, is_av


def load_transformer(
    checkpoint_path: str | Path,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
    preloaded_sd: dict | None = None,
    preloaded_metadata: dict | None = None,
    quantize: str | None = None,
) -> LTXAVModel:
    """Load LTXAVModel from a Kijai separated transformer checkpoint.

    The checkpoint contains FP8 quantized weights (blocks 2+) with per-tensor
    input_scale and weight_scale metadata for scaled matmul. Blocks 0-1 and
    non-attention layers remain in bf16.

    Weights are moved to the target device while preserving their original
    dtypes (fp8 stays fp8, bf16 stays bf16) to fit within 24GB VRAM.

    When ``quantize="int4"``, the FP8 block Linear weights are dequantized
    to bf16 in-place (using the stored per-tensor scales) and the normal
    nn.Linear.forward is kept; later the pipeline applies torchao int4
    weight-only quantization via :func:`quantize_transformer_int4` once the
    model is on GPU. This shrinks the 22B blocks from ~16 GiB FP8 to
    ~5 GiB int4 so the full pipeline fits on a 32 GB GPU without block
    streaming or CPU offload.
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading transformer from: {checkpoint_path}")

    if preloaded_metadata is not None:
        metadata = preloaded_metadata
    else:
        metadata = None
        try:
            from safetensors import safe_open
            with safe_open(str(checkpoint_path), framework="pt") as f:
                metadata = f.metadata()
        except Exception:
            pass

    if preloaded_sd is not None:
        state_dict = preloaded_sd
    else:
        state_dict = load_file(str(checkpoint_path), device="cpu")

    # Kijai checkpoints use "model.diffusion_model." prefix — strip it
    prefix = "model.diffusion_model."
    if any(k.startswith(prefix) for k in list(state_dict.keys())[:5]):
        state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}

    # Separate FP8 scale/quant metadata from model weights
    fp8_suffixes = (".input_scale", ".weight_scale", ".comfy_quant")
    fp8_scales = {}
    model_weights = {}
    for k, v in state_dict.items():
        if any(k.endswith(s) for s in fp8_suffixes):
            fp8_scales[k] = v
        else:
            model_weights[k] = v

    if fp8_scales:
        logger.info(f"Separated {len(fp8_scales)} FP8 scale/quant metadata keys")

    del state_dict

    config, is_av = detect_config_from_state_dict(model_weights, metadata)

    logger.info(f"Instantiating {'LTXAVModel' if is_av else 'LTXVModel'} on meta device...")
    with torch.device("meta"):
        if is_av:
            model = LTXAVModel(**config)
        else:
            from .ltx_model import LTXVModel
            model = LTXVModel(**config)

    logger.info("Loading state dict into model...")
    missing, unexpected = model.load_state_dict(model_weights, strict=False, assign=True)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    del model_weights
    # No GPU allocations happened here — the transformer stays CPU-resident
    # until block streaming. Avoid empty_cache() so we don't contend with a
    # concurrent Gemma GPU transfer happening on a background thread.

    # Attach FP8 per-tensor scales as module attributes in both modes — LoRA
    # merge needs to find them on the module to dequantize/re-quantize
    # correctly, and quantize_transformer_int4 reads them off the module so
    # post-LoRA scale updates propagate.  Only the forward override (scaled
    # matmul) is gated on int4 mode — those blocks get dequantized and
    # re-packed to int4 once they land on GPU.
    if fp8_scales:
        if quantize == "int4":
            attached = attach_fp8_scales(model, fp8_scales)
            logger.info(f"Attached FP8 scales to {attached} linear layers (int4 mode)")
        else:
            patched = patch_fp8_layers(model, fp8_scales)
            logger.info(f"Patched {patched} FP8 linear layers with scaled matmul")
    model._fp8_scales = fp8_scales
    del fp8_scales

    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    logger.info(f"Transformer loaded on CPU: {param_count/1e9:.1f}B params, {mem_gb:.1f}GB")

    return model


def quantize_transformer_int4(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 128,
) -> int:
    """Move transformer to GPU and re-quantize block Linears to int4 in-place.

    Assumes ``load_transformer(quantize="int4")`` left the FP8 block weights
    unpatched (no ``_fp8_scaled_forward`` monkey-patch) with the per-tensor
    weight scales stored on ``model._fp8_scales``.  Working *one block at a
    time* avoids ever holding the full bf16 transformer (~39 GB) on either
    CPU or GPU:

      * move block to GPU (still FP8 for Linear weights)
      * dequantize FP8 Linear weights to bf16 on GPU (per-layer, ≤500 MB)
      * apply torchao ``Int4WeightOnlyConfig`` v1 to that block
      * free the bf16 temporaries via empty_cache

    The scaffold (adaln_single, caption_projection, embedding connectors,
    norms, biases) is moved to GPU in bf16 at the end; scaffold Linears are
    kept at bf16 to preserve numerical sensitivity.

    Returns the number of block Linear layers quantized.
    """
    import gc
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    fp8_scales: dict[str, torch.Tensor] = getattr(model, "_fp8_scales", {})
    _FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # noqa: F841

    def _dequant_fp8_linears_in(module: nn.Module, module_fqn: str) -> None:
        for name, sub in module.named_modules():
            if not isinstance(sub, nn.Linear):
                continue
            if sub.weight.dtype != torch.float8_e4m3fn:
                continue
            # Prefer the module-local scale: LoRA merge updates it in place
            # after merging weights, so reading from model._fp8_scales would
            # be stale.  Fall back to the model-level dict if attach_fp8_scales
            # didn't run (shouldn't happen in practice).
            ws = getattr(sub, "_fp8_weight_scale", None)
            if ws is None:
                full_fqn = f"{module_fqn}.{name}" if name else module_fqn
                scale_key = f"{full_fqn}.weight_scale"
                ws = fp8_scales.get(scale_key)
            if ws is None:
                full_fqn = f"{module_fqn}.{name}" if name else module_fqn
                logger.warning(f"No weight_scale for {full_fqn}; leaving FP8")
                continue
            w = sub.weight.data
            scale = ws.to(device=w.device, dtype=torch.float32)
            sub.weight.data = (w.float() * scale).to(dtype)
            # Strip any previous FP8 patch artifacts if present.
            for attr in ("_fp8_input_scale", "_fp8_weight_scale"):
                if hasattr(sub, attr):
                    delattr(sub, attr)

    block_filter = lambda m, fqn: isinstance(m, nn.Linear)  # noqa: E731

    blocks = model.transformer_blocks
    n_blocks = len(blocks)
    quantized_linears = 0
    t0 = time.time()

    for idx, block in enumerate(blocks):
        # 1. Move this block's tensors to GPU (preserving dtypes).
        for p in block.parameters():
            p.data = p.data.to(device=device, non_blocking=True)
        for b in block.buffers():
            b.data = b.data.to(device=device, non_blocking=True)
        torch.cuda.synchronize(device)

        # 2. Dequantize any FP8 Linear weights to bf16 on GPU.
        _dequant_fp8_linears_in(block, f"transformer_blocks.{idx}")

        # 3. Apply torchao int4 weight-only to every Linear in this block.
        quantize_(block, Int4WeightOnlyConfig(group_size=group_size, version=1), filter_fn=block_filter)
        quantized_linears += sum(1 for m in block.modules() if isinstance(m, nn.Linear))

        # 4. Drop bf16 temporaries from the caching allocator so peak stays
        #    bounded to roughly (current int4 blocks + 1 bf16 block).
        gc.collect()
        torch.cuda.empty_cache()

        if (idx + 1) % 8 == 0 or idx == n_blocks - 1:
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            logger.info(
                f"Quantized block {idx+1}/{n_blocks} to int4 "
                f"({allocated:.2f} GiB allocated)"
            )

    # Scaffold: move everything that isn't a transformer block to GPU.
    scaffold_exclude = {"transformer_blocks"}
    for name, child in model.named_children():
        if name in scaffold_exclude:
            continue
        for p in child.parameters():
            p.data = p.data.to(device=device, non_blocking=True)
        for b in child.buffers():
            b.data = b.data.to(device=device, non_blocking=True)
    for _, p in model.named_parameters(recurse=False):
        p.data = p.data.to(device=device, non_blocking=True)
    for _, b in model.named_buffers(recurse=False):
        b.data = b.data.to(device=device, non_blocking=True)
    torch.cuda.synchronize(device)

    # Drop the stored FP8 scales — they're no longer needed after int4 quant.
    model._fp8_scales = {}

    logger.info(
        f"Transformer int4: {quantized_linears} block Linears quantized, "
        f"scaffold moved to GPU in bf16 ({time.time()-t0:.1f}s)"
    )
    return quantized_linears


def load_vae(
    checkpoint_path: str | Path,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Load a VAE decoder from a separated safetensors checkpoint.

    Works for both video VAE (LTX23_video_vae_bf16.safetensors) and
    audio VAE (LTX23_audio_vae_bf16.safetensors).

    Returns the raw state dict wrapped in a simple namespace since the VAE
    architecture varies. The pipeline will use ltx-core's VAE classes if
    available, or a minimal decoder.
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading VAE from: {checkpoint_path}")

    state_dict = load_file(str(checkpoint_path), device="cpu")

    mem_gb = sum(v.numel() * v.element_size() for v in state_dict.values()) / 1024**3
    logger.info(f"VAE state dict loaded: {len(state_dict)} keys, {mem_gb:.2f}GB")

    return state_dict


def load_text_projection(
    checkpoint_path: str | Path,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Load the text projection weights from a separated safetensors checkpoint.

    Returns the state dict for the text projection model.
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading text projection from: {checkpoint_path}")

    state_dict = load_file(str(checkpoint_path), device=str(device))

    mem_gb = sum(v.numel() * v.element_size() for v in state_dict.values()) / 1024**3
    logger.info(f"Text projection loaded: {len(state_dict)} keys, {mem_gb:.2f}GB")

    return state_dict
