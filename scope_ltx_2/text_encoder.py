"""Text encoding and projection for LTX 2.3.

Implements the V2 text embedding pipeline:
  1. Gemma 3 12B (FP8) produces hidden states from all layers
  2. Per-token RMS normalization across layers
  3. video_aggregate_embed + audio_aggregate_embed project to model dims
  4. Output is concatenated [video_context, audio_context] ready for
     model.preprocess_text_embeds() which runs the embedding connectors.
"""

import logging
import math
import tempfile
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Gemma 3 12B architecture config embedded directly to avoid packaging
# issues with non-Python data files.  Source: google/gemma-3-12b-it config.json
_GEMMA3_CONFIG_DICT: dict = {
    "architectures": ["Gemma3ForConditionalGeneration"],
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
    "eos_token_id": [1, 106],
    "image_token_index": 262144,
    "initializer_range": 0.02,
    "mm_tokens_per_image": 256,
    "model_type": "gemma3",
    "text_config": {
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "model_type": "gemma3_text",
        "num_attention_heads": 16,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
        "sliding_window": 1024,
    },
    "torch_dtype": "bfloat16",
    "vision_config": {
        "hidden_size": 1152,
        "image_size": 896,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
        "vision_use_head": False,
    },
}


def _extract_tokenizer_from_safetensors(fp8_sd: dict[str, torch.Tensor]) -> "GemmaTokenizer":
    """Extract the embedded SentencePiece model from a Comfy-Org FP8 checkpoint
    and return a configured GemmaTokenizer.

    The ``spiece_model`` key stores the raw tokenizer.model bytes as a uint8
    tensor, so no external tokenizer files are needed.
    """
    from transformers import GemmaTokenizer

    sp_tensor = fp8_sd["spiece_model"]
    sp_bytes = sp_tensor.numpy().tobytes()

    tmp = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
    tmp.write(sp_bytes)
    tmp.close()

    tokenizer = GemmaTokenizer(tmp.name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    Path(tmp.name).unlink(missing_ok=True)
    logger.info(f"Extracted tokenizer from spiece_model ({len(sp_bytes)} bytes)")
    return tokenizer


def load_gemma_text_encoder(
    gemma_model_path: str | Path,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    preloaded_sd: dict | None = None,
):
    """Load Gemma 3 12B text encoder, optionally with FP8 weights.

    If gemma_model_path points to a single .safetensors file (e.g. the FP8
    checkpoint from Comfy-Org), the model architecture is constructed from
    the embedded config and the tokenizer is extracted from the spiece_model
    tensor inside the checkpoint.  No external downloads are needed.

    If gemma_model_path points to a directory, it's loaded directly via
    from_pretrained (the directory must contain weights + tokenizer files).

    Args:
        gemma_model_path: Path to FP8 .safetensors file OR model directory
        device: Target device
        dtype: Compute dtype for non-quantized layers

    Returns (model, tokenizer).
    """
    gemma_model_path = Path(gemma_model_path)

    logger.info(f"Loading Gemma 3 12B from: {gemma_model_path}")

    if gemma_model_path.is_file() and gemma_model_path.suffix == ".safetensors":
        logger.info("Loading FP8 Gemma from single safetensors file...")

        from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
        full_config = Gemma3ForConditionalGeneration.config_class.from_dict(_GEMMA3_CONFIG_DICT)
        text_config = full_config.text_config
        with torch.device("meta"):
            model = Gemma3ForCausalLM(text_config)

        fp8_sd = preloaded_sd if preloaded_sd is not None else load_file(str(gemma_model_path), device="cpu")

        tokenizer = _extract_tokenizer_from_safetensors(fp8_sd)

        # Separate FP8 scale metadata from model weights (keep scales for patching)
        fp8_scale_suffixes = (".weight_scale", ".input_scale")
        skip_prefixes = ("vision_model.", "multi_modal_projector.")
        fp8_scales = {}
        model_weights = {}
        n_skipped = 0
        for k, v in fp8_sd.items():
            if any(k.startswith(p) for p in skip_prefixes) or k == "spiece_model":
                n_skipped += 1
            elif any(k.endswith(s) for s in fp8_scale_suffixes):
                fp8_scales[k] = v
            elif k.endswith(".comfy_quant"):
                pass
            else:
                model_weights[k] = v
        fp8_sd.clear()

        logger.info(f"Separated {len(fp8_scales)} FP8 scale keys, skipped {n_skipped} vision/multi_modal keys")

        missing, unexpected = model.load_state_dict(model_weights, strict=False, assign=True)
        del model_weights

        if missing:
            logger.info(f"Missing keys: {len(missing)} (computed buffers: {missing[:3]})")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)}: {unexpected[:5]}...")

        # Re-tie lm_head to embed_tokens (tied weight not in checkpoint)
        if hasattr(model, "lm_head") and model.lm_head.weight.device == torch.device("meta"):
            model.lm_head.weight = model.model.embed_tokens.weight
            logger.info("Re-tied lm_head.weight to embed_tokens.weight")

        # Initialize computed buffers that aren't in the checkpoint.
        # Meta tensors can't use set_data, so we replace via register_buffer
        # on the parent module (same approach as ComfyUI's gemma_encoder.py).
        modules_dict = dict(model.named_modules())
        for name, buf in list(model.named_buffers()):
            if buf.device != torch.device("meta"):
                continue
            parts = name.rsplit(".", 1)
            parent_path, attr = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
            parent = modules_dict[parent_path] if parent_path else model
            if "embed_scale" in attr:
                new_buf = torch.tensor(text_config.hidden_size ** 0.5)
            elif "inv_freq" in attr:
                new_buf = torch.zeros(buf.shape, dtype=torch.float32)
            else:
                new_buf = torch.zeros(buf.shape, dtype=buf.dtype if buf.dtype != torch.float32 else dtype)
            parent.register_buffer(attr, new_buf)
            logger.info(f"Initialized meta buffer: {name}")

        # Patch FP8 linear layers with scaled matmul (same approach as transformer)
        if fp8_scales:
            from .model_loader import patch_fp8_layers
            patched = patch_fp8_layers(model, fp8_scales)
            logger.info(f"Patched {patched} Gemma FP8 linear layers with scaled matmul")
        del fp8_scales

        # Move to device preserving dtypes (FP8 stays FP8).  non_blocking=True
        # lets CUDA batch the per-tensor host→device copies rather than
        # synchronising on every call; this roughly halves the transfer time
        # even for non-pinned (mmap'd) source memory.  A single synchronize
        # at the end ensures all copies complete before the model is used.
        for param in model.parameters():
            param.data = param.data.to(device=device, non_blocking=True)
        for name, buf in model.named_buffers():
            buf.data = buf.data.to(device=device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading from model directory...")
        model = AutoModelForCausalLM.from_pretrained(
            str(gemma_model_path),
            dtype=dtype,
            device_map={"": device} if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device=device, dtype=dtype)

        tokenizer = AutoTokenizer.from_pretrained(str(gemma_model_path))
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    logger.info(f"Gemma loaded: {mem_gb:.1f}GB on {device}")

    return model, tokenizer


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    return x * math.sqrt(target_dim / source_dim)


def _norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm over the hidden dim, then flatten layers.

    Padding tokens must be stripped BEFORE calling this function.

    Args:
        encoded_text: [B, T, D, L] -- hidden states stacked across L layers
                      (only real tokens, no padding)

    Returns:
        [B, T, D*L] normalized and flattened tensor.
    """
    x = encoded_text
    x = x * torch.rsqrt(torch.mean(x ** 2, dim=2, keepdim=True) + 1e-6)
    return x.flatten(start_dim=2)


class TextEmbeddingProjection(nn.Module):
    """V2 feature extractor: per-token RMS norm -> rescale -> aggregate_embed(s).

    Loads video_aggregate_embed and audio_aggregate_embed from the separated
    text projection checkpoint (ltx-2.3_text_projection_bf16.safetensors).
    """

    def __init__(
        self,
        video_aggregate_embed: nn.Linear,
        audio_aggregate_embed: nn.Linear | None,
        embedding_dim: int = 3840,
    ):
        super().__init__()
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        dtype: torch.dtype = torch.bfloat16,
        preloaded_sd: dict[str, torch.Tensor] | None = None,
    ) -> "TextEmbeddingProjection":
        # Use mmap-backed loader so tensors are adopted via assign=True
        # (avoids kaiming_uniform init of 770M-param Linear that would just
        # be overwritten).
        sd = preloaded_sd if preloaded_sd is not None else load_file(str(checkpoint_path), device="cpu")

        prefix = "text_embedding_projection."

        def _load_linear(name: str) -> nn.Linear | None:
            w_key = f"{prefix}{name}.weight"
            b_key = f"{prefix}{name}.bias"
            if w_key not in sd:
                return None
            weight = sd[w_key]
            out_features, in_features = weight.shape
            has_bias = b_key in sd
            # Construct on meta device to skip weight init, then assign loaded
            # tensors directly. The checkpoint is already bf16 so no cast needed.
            with torch.device("meta"):
                linear = nn.Linear(in_features, out_features, bias=has_bias)
            sub_sd = {
                k.removeprefix(f"{prefix}{name}."): v
                for k, v in sd.items()
                if k.startswith(f"{prefix}{name}.")
            }
            linear.load_state_dict(sub_sd, assign=True)
            if linear.weight.dtype != dtype:
                linear = linear.to(dtype=dtype)
            return linear

        video_agg = _load_linear("video_aggregate_embed")
        audio_agg = _load_linear("audio_aggregate_embed")

        if video_agg is None:
            raise ValueError(f"No video_aggregate_embed found in {checkpoint_path}")

        embedding_dim = video_agg.in_features // 49  # 188160 / 49 layers = 3840
        logger.info(
            f"TextEmbeddingProjection loaded: "
            f"video={video_agg.in_features}->{video_agg.out_features}, "
            f"audio={audio_agg.in_features}->{audio_agg.out_features if audio_agg else 'N/A'}, "
            f"embedding_dim={embedding_dim}"
        )
        return cls(video_agg, audio_agg, embedding_dim)

    def forward(
        self,
        all_layer_hiddens: torch.Tensor,
    ) -> torch.Tensor:
        """Project Gemma hidden states to video+audio context.

        Padding tokens must be stripped before calling this method.

        Args:
            all_layer_hiddens: [B, T, D, L] stacked hidden states from all layers
                               (only real tokens, no padding)

        Returns:
            [B, T, video_dim + audio_dim] concatenated context
        """
        normed = _norm_and_concat_per_token_rms(all_layer_hiddens)
        normed = normed.to(self.video_aggregate_embed.weight.dtype)

        v_dim = self.video_aggregate_embed.out_features
        video_ctx = self.video_aggregate_embed(
            _rescale_norm(normed, v_dim, self.embedding_dim)
        )

        if self.audio_aggregate_embed is not None:
            a_dim = self.audio_aggregate_embed.out_features
            audio_ctx = self.audio_aggregate_embed(
                _rescale_norm(normed, a_dim, self.embedding_dim)
            )
            return torch.cat([video_ctx, audio_ctx], dim=-1)

        return video_ctx


def encode_prompt(
    text_encoder,
    tokenizer,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 512,
) -> torch.Tensor:
    """Encode a text prompt using Gemma and return stacked hidden states.

    Tokenizes plain text (no chat template) with left-padding, then strips
    padding tokens so the output contains only real tokens.  This matches
    ComfyUI's encode_token_weights flow where raw text is fed directly.

    Returns:
        all_layer_hiddens: [B, T_real, D, L] hidden states from all transformer layers
                           with padding stripped (only real tokens).
    """
    inputs = tokenizer(
        prompt.strip(),
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).to(device)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        outputs = text_encoder(
            **inputs,
            output_hidden_states=True,
        )

    # outputs.hidden_states includes the embedding layer + all transformer layers.
    # Use ALL of them (embedding + 48 transformer = 49 total) to match the
    # aggregate_embed input dim of 188160 = 3840 * 49.
    all_layer_hiddens = torch.stack(list(outputs.hidden_states), dim=-1).to(dtype=dtype)

    # Strip left-padding tokens: only keep real tokens.
    # With left-padding, real tokens are at the END of the sequence.
    attention_mask = inputs["attention_mask"]
    real_token_count = int(attention_mask.sum().item())
    all_layer_hiddens = all_layer_hiddens[:, -real_token_count:, :, :]

    logger.info(
        f"Encoded prompt: {all_layer_hiddens.shape[3]} layers, "
        f"dim={all_layer_hiddens.shape[2]}, "
        f"seq_len={all_layer_hiddens.shape[1]} (stripped from {attention_mask.shape[1]})"
    )

    return all_layer_hiddens
