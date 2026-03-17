"""Text encoding and projection for LTX 2.3.

Implements the V2 text embedding pipeline:
  1. Gemma 3 12B produces hidden states from all layers
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
    gemma_path: str | Path,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
):
    """Load Gemma 3 12B text encoder using transformers.

    Returns the model and tokenizer. The model is loaded in the specified dtype
    and placed on the given device.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gemma_path = str(gemma_path)
    logger.info(f"Loading Gemma 3 12B from: {gemma_path}")

    tokenizer = AutoTokenizer.from_pretrained(gemma_path)

    model = AutoModelForCausalLM.from_pretrained(
        gemma_path,
        dtype=dtype,
        device_map={"": device} if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device=device, dtype=dtype)
    model.eval()

    param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    logger.info(f"Gemma loaded: {param_gb:.1f}GB on {device}")

    return model, tokenizer


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    return x * math.sqrt(target_dim / source_dim)


def _norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm over the hidden dim, then flatten layers.

    Args:
        encoded_text: [B, T, D, L] — hidden states stacked across L layers
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
    """V2 feature extractor: per-token RMS norm → rescale → aggregate_embed(s).

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
            f"video={video_agg.in_features}→{video_agg.out_features}, "
            f"audio={audio_agg.in_features}→{audio_agg.out_features if audio_agg else 'N/A'}, "
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

    with torch.no_grad():
        outputs = text_encoder(
            **inputs,
            output_hidden_states=True,
        )

    # outputs.hidden_states is a tuple of (num_layers+1) tensors [B, T, D]
    # Skip the embedding layer (index 0), use all transformer layer outputs
    hidden_states_list = outputs.hidden_states[1:]
    all_layer_hiddens = torch.stack(hidden_states_list, dim=-1).to(dtype=dtype)
    # all_layer_hiddens: [B, T, D, L]

    attention_mask = inputs["attention_mask"]

    logger.info(
        f"Encoded prompt: {all_layer_hiddens.shape[3]} layers, "
        f"dim={all_layer_hiddens.shape[2]}, seq_len={all_layer_hiddens.shape[1]}"
    )

    return all_layer_hiddens, attention_mask
