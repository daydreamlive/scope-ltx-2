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
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def load_gemma_text_encoder(
    gemma_model_path: str | Path,
    tokenizer_path: str | Path,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    """Load Gemma 3 12B text encoder, optionally with FP8 weights.

    If gemma_model_path points to a single .safetensors file (e.g. the FP8
    checkpoint from Comfy-Org), the model structure is loaded from
    tokenizer_path (which must contain config.json) and then the FP8 weights
    are loaded on top, preserving their dtypes.

    If gemma_model_path points to a directory, it's loaded directly via
    from_pretrained.

    Args:
        gemma_model_path: Path to FP8 .safetensors file OR model directory
        tokenizer_path: Path to directory with config.json + tokenizer files
        device: Target device
        dtype: Compute dtype for non-quantized layers

    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gemma_model_path = Path(gemma_model_path)
    tokenizer_path = Path(tokenizer_path)

    logger.info(f"Loading Gemma 3 12B from: {gemma_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if gemma_model_path.is_file() and gemma_model_path.suffix == ".safetensors":
        logger.info("Loading FP8 Gemma from single safetensors file...")

        # The Comfy-Org FP8 checkpoint contains a full Gemma3ForConditionalGeneration
        # (text + vision) but we only need the language model. The checkpoint keys
        # for the language model are stored as model.layers.*, model.embed_tokens.*
        # which match Gemma3ForCausalLM exactly. Vision/multi_modal keys are skipped.
        from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
        full_config = Gemma3ForConditionalGeneration.config_class.from_pretrained(str(tokenizer_path))
        text_config = full_config.text_config
        with torch.device("meta"):
            model = Gemma3ForCausalLM(text_config)

        fp8_sd = load_file(str(gemma_model_path), device="cpu")

        # Strip FP8 scale/quant metadata and non-language keys
        fp8_suffixes = (".weight_scale", ".input_scale", ".comfy_quant")
        skip_prefixes = ("vision_model.", "multi_modal_projector.")
        model_weights = {}
        n_scales = 0
        n_skipped = 0
        for k, v in fp8_sd.items():
            if any(k.endswith(s) for s in fp8_suffixes):
                n_scales += 1
            elif any(k.startswith(p) for p in skip_prefixes) or k == "spiece_model":
                n_skipped += 1
            else:
                model_weights[k] = v
        del fp8_sd

        logger.info(f"Stripped {n_scales} FP8 scale/quant keys, skipped {n_skipped} vision/multi_modal keys")

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

        # Move to device preserving dtypes (FP8 stays FP8)
        for param in model.parameters():
            param.data = param.data.to(device=device)
        for name, buf in model.named_buffers():
            buf.data = buf.data.to(device=device)

    else:
        logger.info("Loading from model directory...")
        model = AutoModelForCausalLM.from_pretrained(
            str(gemma_model_path),
            dtype=dtype,
            device_map={"": device} if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device=device, dtype=dtype)

    model.eval()

    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    logger.info(f"Gemma loaded: {mem_gb:.1f}GB on {device}")

    return model, tokenizer


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    return x * math.sqrt(target_dim / source_dim)


def _norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm over the hidden dim, then flatten layers.

    Args:
        encoded_text: [B, T, D, L] -- hidden states stacked across L layers
        attention_mask: [B, T] binary mask (1=real, 0=pad)

    Returns:
        [B, T, D*L] normalized and flattened tensor with padding zeroed out.
    """
    B, T, D, L = encoded_text.shape
    variance = torch.mean(encoded_text ** 2, dim=2, keepdim=True)
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)
    normed = torch.where(mask_3d, normed, torch.zeros_like(normed))
    return normed


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
    ) -> "TextEmbeddingProjection":
        sd = load_file(str(checkpoint_path), device="cpu")

        prefix = "text_embedding_projection."

        def _load_linear(name: str) -> nn.Linear | None:
            w_key = f"{prefix}{name}.weight"
            b_key = f"{prefix}{name}.bias"
            if w_key not in sd:
                return None
            weight = sd[w_key]
            out_features, in_features = weight.shape
            has_bias = b_key in sd
            linear = nn.Linear(in_features, out_features, bias=has_bias)
            sub_sd = {}
            for k, v in sd.items():
                if k.startswith(f"{prefix}{name}."):
                    sub_sd[k.removeprefix(f"{prefix}{name}.")] = v
            linear.load_state_dict(sub_sd)
            return linear.to(dtype=dtype)

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
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project Gemma hidden states to video+audio context.

        Args:
            all_layer_hiddens: [B, T, D, L] stacked hidden states from all layers
            attention_mask: [B, T] binary mask

        Returns:
            [B, T, video_dim + audio_dim] concatenated context
        """
        normed = _norm_and_concat_per_token_rms(all_layer_hiddens, attention_mask)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a text prompt using Gemma and return stacked hidden states.

    Returns:
        all_layer_hiddens: [B, T, D, L] hidden states from all transformer layers
        attention_mask: [B, T] binary mask
    """
    inputs = tokenizer(
        prompt,
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

    attention_mask = inputs["attention_mask"]

    logger.info(
        f"Encoded prompt: {all_layer_hiddens.shape[3]} layers, "
        f"dim={all_layer_hiddens.shape[2]}, seq_len={all_layer_hiddens.shape[1]}"
    )

    return all_layer_hiddens, attention_mask
