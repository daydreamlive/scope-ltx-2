"""Standalone LTX 2.3 transformer model adapted from ComfyUI's lightricks implementation.
All comfy.* dependencies have been replaced with inline equivalents."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple
import functools
import logging
import math

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .patchifier import SymmetricPatchifier, AudioPatchifier, latent_to_pixel_coords

logger = logging.getLogger(__name__)


def _log_base(x, base):
    return np.log(x) / np.log(base)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            norm = norm * self.weight
        return norm


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class LTXRopeType(str, Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"

    KEY = "rope_type"

    @classmethod
    def from_dict(cls, kwargs, default=None):
        if default is None:
            default = cls.INTERLEAVED
        return cls(kwargs.get(cls.KEY, default))


class LTXFrequenciesPrecision(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    KEY = "frequencies_precision"

    @classmethod
    def from_dict(cls, kwargs, default=None):
        if default is None:
            default = cls.FLOAT32
        return cls(kwargs.get(cls.KEY, default))


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
        dtype=None,
        device=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias, dtype=dtype, device=device)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False, dtype=dtype, device=device)
        else:
            self.cond_proj = None

        self.act = nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(
            time_embed_dim, time_embed_dim_out, sample_proj_bias, dtype=dtype, device=device
        )

        if post_act_fn is None:
            self.post_act = None

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim,
        size_emb_dim,
        use_additional_conditions: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype, device=device
        )

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb


class AdaLayerNormSingle(nn.Module):
    def __init__(
        self, embedding_dim: int, embedding_coefficient: int = 6, use_additional_conditions: bool = False, dtype=None, device=None
    ):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
            dtype=dtype,
            device=device,
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class PixArtAlphaTextProjection(nn.Module):
    def __init__(
        self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh", dtype=None, device=None
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(
            in_features=in_features, out_features=hidden_size, bias=True, dtype=dtype, device=device
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(
            in_features=hidden_size, out_features=out_features, bias=True, dtype=dtype, device=device
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class NormSingleLinearTextProjection(nn.Module):
    def __init__(
        self, in_features, hidden_size, dtype=None, device=None
    ):
        super().__init__()
        self.in_norm = RMSNorm(in_features, eps=1e-6, elementwise_affine=False)
        self.linear_1 = nn.Linear(
            in_features, hidden_size, bias=True, dtype=dtype, device=device
        )
        self.hidden_size = hidden_size
        self.in_features = in_features

    def forward(self, caption):
        caption = self.in_norm(caption)
        caption = caption * (self.hidden_size / self.in_features) ** 0.5
        return self.linear_1(caption)


class GELU_approx(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, dtype=dtype, device=device)

    def forward(self, x):
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out, mult=4, glu=False, dropout=0.0, dtype=None, device=None):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELU_approx(dim, inner_dim, dtype=dtype, device=device)

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)


def apply_rotary_emb(input_tensor, freqs_cis):
    cos_freqs, sin_freqs = freqs_cis[0], freqs_cis[1]
    split_pe = freqs_cis[2] if len(freqs_cis) > 2 else False
    return (
        apply_split_rotary_emb(input_tensor, cos_freqs, sin_freqs)
        if split_pe else
        apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    )


def apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs):
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

    return out


def apply_split_rotary_emb(input_tensor, cos, sin):
    needs_reshape = False
    if input_tensor.ndim != 4 and cos.ndim == 4:
        B, H, T, _ = cos.shape
        input_tensor = input_tensor.reshape(B, T, H, -1).swapaxes(1, 2)
        needs_reshape = True
    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]
    output = split_input * cos.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]
    first_half_output.addcmul_(-sin.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin.unsqueeze(-2), first_half_input)
    output = rearrange(output, "... d r -> ... (d r)")
    return output.swapaxes(1, 2).reshape(B, T, -1) if needs_reshape else output


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_precision=None,
        apply_gated_attention=False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.attn_precision = attn_precision

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = RMSNorm(inner_dim, eps=1e-5)
        self.k_norm = RMSNorm(inner_dim, eps=1e-5)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True, dtype=dtype, device=device)

        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True, dtype=dtype, device=device)
        else:
            self.to_gate_logits = None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout)
        )

    def _attention(self, q, k, v, mask=None):
        heads = self.heads
        b, seq_len, dim = q.shape
        head_dim = dim // heads
        q = q.view(b, seq_len, heads, head_dim).transpose(1, 2)
        k = k.view(b, k.shape[1], heads, head_dim).transpose(1, 2)
        v = v.view(b, v.shape[1], heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, seq_len, dim)
        return out

    def forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

        out = self._attention(q, k, v, mask=mask)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)


ADALN_BASE_PARAMS_COUNT = 6
ADALN_CROSS_ATTN_PARAMS_COUNT = 9


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim, n_heads, d_head, context_dim=None, attn_precision=None, cross_attention_adaln=False, dtype=None, device=None
    ):
        super().__init__()

        self.attn_precision = attn_precision
        self.cross_attention_adaln = cross_attention_adaln
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            context_dim=None,
            attn_precision=self.attn_precision,
            dtype=dtype,
            device=device,
        )
        self.ff = FeedForward(dim, dim_out=dim, glu=True, dtype=dtype, device=device)

        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            attn_precision=self.attn_precision,
            dtype=dtype,
            device=device,
        )

        num_ada_params = ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.scale_shift_table = nn.Parameter(torch.empty(num_ada_params, dim, device=device, dtype=dtype))

        if cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, dim, device=device, dtype=dtype))

    def forward(self, x, context=None, attention_mask=None, timestep=None, pe=None, transformer_options={}, self_attention_mask=None, prompt_timestep=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None, None, :6].to(device=x.device, dtype=x.dtype) + timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)[:, :, :6, :]).unbind(dim=2)

        x += self.attn1(rms_norm(x) * (1 + scale_msa) + shift_msa, pe=pe, mask=self_attention_mask, transformer_options=transformer_options) * gate_msa

        if self.cross_attention_adaln:
            shift_q_mca, scale_q_mca, gate_mca = (self.scale_shift_table[None, None, 6:9].to(device=x.device, dtype=x.dtype) + timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)[:, :, 6:9, :]).unbind(dim=2)
            x += apply_cross_attention_adaln(
                x, context, self.attn2, shift_q_mca, scale_q_mca, gate_mca,
                self.prompt_scale_shift_table, prompt_timestep, attention_mask, transformer_options,
            )
        else:
            x += self.attn2(x, context=context, mask=attention_mask, transformer_options=transformer_options)

        y = rms_norm(x)
        y = torch.addcmul(y, y, scale_mlp).add_(shift_mlp)
        x.addcmul_(self.ff(y), gate_mlp)

        return x


def compute_prompt_timestep(adaln_module, timestep_scaled, batch_size, hidden_dtype):
    if adaln_module is None:
        return None
    ts_input = (
        timestep_scaled.max(dim=1, keepdim=True).values.flatten()
        if timestep_scaled.dim() > 1
        else timestep_scaled.flatten()
    )
    prompt_ts, _ = adaln_module(
        ts_input,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )
    return prompt_ts.view(batch_size, 1, prompt_ts.shape[-1])


def apply_cross_attention_adaln(
    x, context, attn, q_shift, q_scale, q_gate,
    prompt_scale_shift_table, prompt_timestep,
    attention_mask=None, transformer_options={},
):
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=attention_mask, transformer_options=transformer_options) * q_gate


def get_fractional_positions(indices_grid, max_pos):
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), f'Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})'
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        axis=-1,
    )
    return fractional_positions


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(positional_embedding_theta, positional_embedding_max_pos_count, inner_dim, _=None):
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            _log_base(start, theta),
            _log_base(end, theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


def generate_freq_grid_pytorch(positional_embedding_theta, positional_embedding_max_pos_count, inner_dim, device):
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            device=device,
            dtype=torch.float32,
        )
    )
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2

    return indices


def generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid):
    if use_middle_indices_grid:
        assert(len(indices_grid.shape) == 4 and indices_grid.shape[-1] == 2)
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    freqs = (
        (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
        .transpose(-1, -2)
        .flatten(2)
    )
    return freqs


def interleaved_freqs_cis(freqs, pad_size):
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def split_freqs_cis(freqs, pad_size, num_attention_heads):
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    B, T, half_HD = cos_freq.shape

    cos_freq = cos_freq.reshape(B, T, num_attention_heads, half_HD // num_attention_heads)
    sin_freq = sin_freq.reshape(B, T, num_attention_heads, half_HD // num_attention_heads)

    cos_freq = torch.swapaxes(cos_freq, 1, 2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)
    return cos_freq, sin_freq


class LTXBaseModel(torch.nn.Module, ABC):
    def __init__(
        self,
        in_channels: int,
        cross_attention_dim: int,
        attention_head_dim: int,
        num_attention_heads: int,
        caption_channels: int,
        num_layers: int,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list = [20, 2048, 2048],
        causal_temporal_positioning: bool = False,
        vae_scale_factors: tuple = (8, 32, 32),
        use_middle_indices_grid=False,
        timestep_scale_multiplier=1000.0,
        caption_proj_before_connector=False,
        cross_attention_adaln=False,
        caption_projection_first_linear=True,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.generator = None
        self.vae_scale_factors = vae_scale_factors
        self.use_middle_indices_grid = use_middle_indices_grid
        self.dtype = dtype
        self.in_channels = in_channels
        self.cross_attention_dim = cross_attention_dim
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.caption_channels = caption_channels
        self.num_layers = num_layers
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.split_positional_embedding = LTXRopeType.from_dict(kwargs)
        self.freq_grid_generator = (
            generate_freq_grid_np if LTXFrequenciesPrecision.from_dict(kwargs) == LTXFrequenciesPrecision.FLOAT64
            else generate_freq_grid_pytorch
        )
        self.causal_temporal_positioning = causal_temporal_positioning
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.caption_proj_before_connector = caption_proj_before_connector
        self.cross_attention_adaln = cross_attention_adaln
        self.caption_projection_first_linear = caption_projection_first_linear

        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels

        self._init_common_components(device, dtype)
        self._init_model_components(device, dtype, **kwargs)
        self._init_transformer_blocks(device, dtype, **kwargs)
        self._init_output_components(device, dtype)

    def _init_common_components(self, device, dtype):
        self.patchify_proj = nn.Linear(
            self.in_channels, self.inner_dim, bias=True, dtype=dtype, device=device
        )

        embedding_coefficient = ADALN_CROSS_ATTN_PARAMS_COUNT if self.cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=embedding_coefficient, use_additional_conditions=False, dtype=dtype, device=device
        )

        if self.cross_attention_adaln:
            self.prompt_adaln_single = AdaLayerNormSingle(
                self.inner_dim, embedding_coefficient=2, use_additional_conditions=False, dtype=dtype, device=device
            )
        else:
            self.prompt_adaln_single = None

        if self.caption_proj_before_connector:
            if self.caption_projection_first_linear:
                self.caption_projection = NormSingleLinearTextProjection(
                    in_features=self.caption_channels,
                    hidden_size=self.inner_dim,
                    dtype=dtype,
                    device=device,
                )
            else:
                self.caption_projection = lambda a: a
        else:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels,
                hidden_size=self.inner_dim,
                dtype=dtype,
                device=device,
            )

    @abstractmethod
    def _init_model_components(self, device, dtype, **kwargs):
        pass

    @abstractmethod
    def _init_transformer_blocks(self, device, dtype, **kwargs):
        pass

    @abstractmethod
    def _init_output_components(self, device, dtype):
        pass

    @abstractmethod
    def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        pass

    @abstractmethod
    def _process_transformer_blocks(self, x, context, attention_mask, timestep, pe, self_attention_mask=None, **kwargs):
        pass

    @abstractmethod
    def _process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        pass

    def _prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        grid_mask = kwargs.get("grid_mask", None)
        if grid_mask is not None:
            timestep = timestep[:, grid_mask]

        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep_emb, embedded_timestep = self.adaln_single(
            timestep_scaled.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )

        timestep_emb = timestep_emb.view(batch_size, -1, timestep_emb.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        prompt_timestep = compute_prompt_timestep(
            self.prompt_adaln_single, timestep_scaled, batch_size, hidden_dtype
        )

        return timestep_emb, embedded_timestep, prompt_timestep

    def _prepare_context(self, context, batch_size, x, attention_mask=None):
        if self.caption_proj_before_connector is False:
            context = self.caption_projection(context)

        context = context.view(batch_size, -1, x.shape[-1])
        return context, attention_mask

    def _precompute_freqs_cis(
        self,
        indices_grid,
        dim,
        out_dtype,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        use_middle_indices_grid=False,
        num_attention_heads=32,
    ):
        split_mode = self.split_positional_embedding == LTXRopeType.SPLIT
        indices = self.freq_grid_generator(theta, indices_grid.shape[1], dim, indices_grid.device)
        freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

        if split_mode:
            expected_freqs = dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
        else:
            n_elem = 2 * indices_grid.shape[1]
            cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
        return cos_freq.to(out_dtype), sin_freq.to(out_dtype), split_mode

    def _prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)
        pe = self._precompute_freqs_cis(
            fractional_coords,
            dim=self.inner_dim,
            out_dtype=x_dtype,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
        )
        return pe

    def _prepare_attention_mask(self, attention_mask, x_dtype):
        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x_dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ) * torch.finfo(x_dtype).max
        return attention_mask

    def forward(
        self, x, timestep, context, attention_mask, frame_rate=25, transformer_options={}, keyframe_idxs=None, denoise_mask=None, **kwargs
    ):
        return self._forward(x, timestep, context, attention_mask, frame_rate, transformer_options, keyframe_idxs, denoise_mask=denoise_mask, **kwargs)

    def _forward(
        self, x, timestep, context, attention_mask, frame_rate=25, transformer_options={}, keyframe_idxs=None, denoise_mask=None, **kwargs
    ):
        if isinstance(x, list):
            input_dtype = x[0].dtype
            batch_size = x[0].shape[0]
        else:
            input_dtype = x.dtype
            batch_size = x.shape[0]

        merged_args = {**transformer_options, **kwargs}
        x, pixel_coords, additional_args = self._process_input(x, keyframe_idxs, denoise_mask, **merged_args)
        merged_args.update(additional_args)

        timestep_emb, embedded_timestep, prompt_timestep = self._prepare_timestep(timestep, batch_size, input_dtype, **merged_args)
        merged_args["prompt_timestep"] = prompt_timestep
        context, attention_mask = self._prepare_context(context, batch_size, x, attention_mask)

        attention_mask = self._prepare_attention_mask(attention_mask, input_dtype)
        pe = self._prepare_positional_embeddings(pixel_coords, frame_rate, input_dtype)

        x = self._process_transformer_blocks(
            x, context, attention_mask, timestep_emb, pe,
            transformer_options=transformer_options,
            **merged_args,
        )

        x = self._process_output(x, embedded_timestep, keyframe_idxs, **merged_args)
        return x


class LTXVModel(LTXBaseModel):
    def __init__(
        self,
        in_channels=128,
        cross_attention_dim=2048,
        attention_head_dim=64,
        num_attention_heads=32,
        caption_channels=4096,
        num_layers=28,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        causal_temporal_positioning=False,
        vae_scale_factors=(8, 32, 32),
        use_middle_indices_grid=False,
        timestep_scale_multiplier=1000.0,
        caption_proj_before_connector=False,
        cross_attention_adaln=False,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            caption_channels=caption_channels,
            num_layers=num_layers,
            positional_embedding_theta=positional_embedding_theta,
            positional_embedding_max_pos=positional_embedding_max_pos,
            causal_temporal_positioning=causal_temporal_positioning,
            vae_scale_factors=vae_scale_factors,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            caption_proj_before_connector=caption_proj_before_connector,
            cross_attention_adaln=cross_attention_adaln,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    def _init_model_components(self, device, dtype, **kwargs):
        pass

    def _init_transformer_blocks(self, device, dtype, **kwargs):
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    context_dim=self.cross_attention_dim,
                    cross_attention_adaln=self.cross_attention_adaln,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _init_output_components(self, device, dtype):
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim, dtype=dtype, device=device))
        self.norm_out = nn.LayerNorm(
            self.inner_dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, dtype=dtype, device=device)
        self.patchifier = SymmetricPatchifier(1, start_end=True)

    def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        additional_args = {"orig_shape": list(x.shape)}
        x, latent_coords = self.patchifier.patchify(x)
        pixel_coords = latent_to_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.vae_scale_factors,
            causal_fix=self.causal_temporal_positioning,
        )

        grid_mask = None
        if keyframe_idxs is not None:
            additional_args.update({"orig_patchified_shape": list(x.shape)})
            denoise_mask = self.patchifier.patchify(denoise_mask)[0]
            grid_mask = ~torch.any(denoise_mask < 0, dim=-1)[0]
            additional_args.update({"grid_mask": grid_mask})
            x = x[:, grid_mask, :]
            pixel_coords = pixel_coords[:, :, grid_mask, ...]

            kf_grid_mask = grid_mask[-keyframe_idxs.shape[2]:]
            keyframe_idxs = keyframe_idxs[..., kf_grid_mask, :]
            pixel_coords[:, :, -keyframe_idxs.shape[2]:, :] = keyframe_idxs
            additional_args["num_guide_tokens"] = keyframe_idxs.shape[2]

        x = self.patchify_proj(x)
        return x, pixel_coords, additional_args

    def _process_transformer_blocks(self, x, context, attention_mask, timestep, pe, transformer_options={}, self_attention_mask=None, **kwargs):
        prompt_timestep = kwargs.get("prompt_timestep", None)

        for block in self.transformer_blocks:
            x = block(
                x,
                context=context,
                attention_mask=attention_mask,
                timestep=timestep,
                pe=pe,
                transformer_options=transformer_options,
                self_attention_mask=self_attention_mask,
                prompt_timestep=prompt_timestep,
            )

        return x

    def _process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        scale_shift_values = (
            self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = self.norm_out(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)

        if keyframe_idxs is not None:
            grid_mask = kwargs["grid_mask"]
            orig_patchified_shape = kwargs["orig_patchified_shape"]
            full_x = torch.zeros(orig_patchified_shape, dtype=x.dtype, device=x.device)
            full_x[:, grid_mask, :] = x
            x = full_x

        orig_shape = kwargs["orig_shape"]
        x = self.patchifier.unpatchify(
            latents=x,
            output_height=orig_shape[3],
            output_width=orig_shape[4],
            output_num_frames=orig_shape[2],
            out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size),
        )

        return x


class BasicTransformerBlock1D(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        context_dim=None,
        attn_precision=None,
        apply_gated_attention=False,
        dtype=None,
        device=None,
    ):
        super().__init__()

        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            context_dim=None,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
        )

        self.ff = FeedForward(
            dim,
            dim_out=dim,
            glu=True,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states, attention_mask=None, pe=None) -> torch.FloatTensor:
        norm_hidden_states = rms_norm(hidden_states)
        norm_hidden_states = norm_hidden_states.squeeze(1)

        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = rms_norm(hidden_states)

        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels=128,
        cross_attention_dim=2048,
        attention_head_dim=128,
        num_attention_heads=30,
        num_layers=2,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4096],
        causal_temporal_positioning=False,
        num_learnable_registers: Optional[int] = 128,
        apply_gated_attention=False,
        dtype=None,
        device=None,
        split_rope=False,
        double_precision_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.out_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.split_rope = split_rope
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = nn.ModuleList(
            [
                BasicTransformerBlock1D(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    context_dim=cross_attention_dim,
                    apply_gated_attention=apply_gated_attention,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = nn.Parameter(
                torch.empty(
                    self.num_learnable_registers, inner_dim, dtype=dtype, device=device
                )
            )

    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack(
            [
                indices_grid[:, i] / self.positional_embedding_max_pos[i]
                for i in range(1)
            ],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs(self, indices_grid, spacing):
        source_dtype = indices_grid.dtype
        dtype = (
            torch.float32
            if source_dtype in (torch.bfloat16, torch.float16)
            else source_dtype
        )

        fractional_positions = self.get_fractional_positions(indices_grid)
        indices = (
            generate_freq_grid_np(
                self.positional_embedding_theta,
                indices_grid.shape[1],
                self.inner_dim,
            )
            if self.double_precision_rope
            else self.generate_freq_grid(spacing, dtype, fractional_positions.device)
        ).to(device=fractional_positions.device)

        if spacing == "exp_2":
            freqs = (
                (indices * fractional_positions.unsqueeze(-1))
                .transpose(-1, -2)
                .flatten(2)
            )
        else:
            freqs = (
                (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
                .transpose(-1, -2)
                .flatten(2)
            )
        return freqs

    def generate_freq_grid(self, spacing, dtype, device):
        dim = self.inner_dim
        theta = self.positional_embedding_theta
        n_pos_dims = 1
        n_elem = 2 * n_pos_dims
        start = 1
        end = theta

        if spacing == "exp":
            indices = theta ** (torch.arange(0, dim, n_elem, device="cpu", dtype=torch.float32) / (dim - n_elem))
            indices = indices.to(dtype=dtype, device=device)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, n_elem, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(
                start, end, dim // n_elem, device=device, dtype=dtype
            )
        elif spacing == "sqrt":
            indices = torch.linspace(
                start**2, end**2, dim // n_elem, device=device, dtype=dtype
            ).sqrt()

        indices = indices * math.pi / 2

        return indices

    def precompute_freqs_cis(self, indices_grid, spacing="exp", out_dtype=None):
        dim = self.inner_dim
        n_elem = 2
        freqs = self.precompute_freqs(indices_grid, spacing)
        if self.split_rope:
            expected_freqs = dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq, sin_freq = split_freqs_cis(
                freqs, pad_size, self.num_attention_heads
            )
        else:
            cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
        return cos_freq.to(dtype=out_dtype), sin_freq.to(dtype=out_dtype), self.split_rope

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if self.num_learnable_registers:
            num_registers_duplications = math.ceil(
                max(1024, hidden_states.shape[1]) / self.num_learnable_registers
            )
            learnable_registers = torch.tile(
                self.learnable_registers.to(hidden_states), (num_registers_duplications, 1)
            )

            hidden_states = torch.cat((hidden_states, learnable_registers[hidden_states.shape[1]:].unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)), dim=1)

            if attention_mask is not None:
                attention_mask = torch.zeros([1, 1, 1, hidden_states.shape[1]], dtype=attention_mask.dtype, device=attention_mask.device)

        indices_grid = torch.arange(
            hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device
        )
        indices_grid = indices_grid[None, None, :]
        freqs_cis = self.precompute_freqs_cis(indices_grid, out_dtype=hidden_states.dtype)

        for block in self.transformer_1d_blocks:
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, pe=freqs_cis
            )

        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask


class CompressedTimestep:
    __slots__ = ('data', 'batch_size', 'num_frames', 'patches_per_frame', 'feature_dim')

    def __init__(self, tensor: torch.Tensor, patches_per_frame: int):
        self.batch_size, num_tokens, self.feature_dim = tensor.shape

        if patches_per_frame is not None and num_tokens % patches_per_frame == 0 and num_tokens >= patches_per_frame:
            self.patches_per_frame = patches_per_frame
            self.num_frames = num_tokens // patches_per_frame

            reshaped = tensor.view(self.batch_size, self.num_frames, patches_per_frame, self.feature_dim)
            self.data = reshaped[:, :, 0, :].contiguous()
        else:
            self.patches_per_frame = 1
            self.num_frames = num_tokens
            self.data = tensor

    def expand(self):
        if self.patches_per_frame == 1:
            return self.data

        expanded = self.data.unsqueeze(2).expand(self.batch_size, self.num_frames, self.patches_per_frame, self.feature_dim)
        return expanded.reshape(self.batch_size, -1, self.feature_dim)

    def expand_for_computation(self, scale_shift_table: torch.Tensor, batch_size: int, indices: slice = slice(None, None)):
        num_ada_params = scale_shift_table.shape[0]

        if self.patches_per_frame == 1:
            num_tokens = self.data.shape[1]
            dim_per_param = self.feature_dim // num_ada_params
            reshaped = self.data.reshape(batch_size, num_tokens, num_ada_params, dim_per_param)[:, :, indices, :]
            table_values = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=self.data.device, dtype=self.data.dtype)
            ada_values = (table_values + reshaped).unbind(dim=2)
            return ada_values

        frame_reshaped = self.data.reshape(batch_size, self.num_frames, num_ada_params, -1)[:, :, indices, :]
        table_values = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(
            device=self.data.device, dtype=self.data.dtype
        )
        frame_ada = (table_values + frame_reshaped).unbind(dim=2)

        return tuple(
            frame_val.unsqueeze(2).expand(batch_size, self.num_frames, self.patches_per_frame, -1)
            .reshape(batch_size, -1, frame_val.shape[-1])
            for frame_val in frame_ada
        )


class BasicAVTransformerBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        a_dim,
        v_heads,
        a_heads,
        vd_head,
        ad_head,
        v_context_dim=None,
        a_context_dim=None,
        attn_precision=None,
        apply_gated_attention=False,
        cross_attention_adaln=False,
        dtype=None,
        device=None,
    ):
        super().__init__()

        self.attn_precision = attn_precision
        self.cross_attention_adaln = cross_attention_adaln

        self.attn1 = CrossAttention(
            query_dim=v_dim, heads=v_heads, dim_head=vd_head, context_dim=None,
            attn_precision=self.attn_precision, apply_gated_attention=apply_gated_attention,
            dtype=dtype, device=device,
        )
        self.audio_attn1 = CrossAttention(
            query_dim=a_dim, heads=a_heads, dim_head=ad_head, context_dim=None,
            attn_precision=self.attn_precision, apply_gated_attention=apply_gated_attention,
            dtype=dtype, device=device,
        )

        self.attn2 = CrossAttention(
            query_dim=v_dim, context_dim=v_context_dim, heads=v_heads, dim_head=vd_head,
            attn_precision=self.attn_precision, apply_gated_attention=apply_gated_attention,
            dtype=dtype, device=device,
        )
        self.audio_attn2 = CrossAttention(
            query_dim=a_dim, context_dim=a_context_dim, heads=a_heads, dim_head=ad_head,
            attn_precision=self.attn_precision, apply_gated_attention=apply_gated_attention,
            dtype=dtype, device=device,
        )

        self.audio_to_video_attn = CrossAttention(
            query_dim=v_dim, context_dim=a_dim, heads=a_heads, dim_head=ad_head,
            attn_precision=self.attn_precision, apply_gated_attention=apply_gated_attention,
            dtype=dtype, device=device,
        )

        self.video_to_audio_attn = CrossAttention(
            query_dim=a_dim, context_dim=v_dim, heads=a_heads, dim_head=ad_head,
            attn_precision=self.attn_precision, apply_gated_attention=apply_gated_attention,
            dtype=dtype, device=device,
        )

        self.ff = FeedForward(
            v_dim, dim_out=v_dim, glu=True, dtype=dtype, device=device
        )
        self.audio_ff = FeedForward(
            a_dim, dim_out=a_dim, glu=True, dtype=dtype, device=device
        )

        num_ada_params = ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.scale_shift_table = nn.Parameter(torch.empty(num_ada_params, v_dim, device=device, dtype=dtype))
        self.audio_scale_shift_table = nn.Parameter(
            torch.empty(num_ada_params, a_dim, device=device, dtype=dtype)
        )

        if cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, v_dim, device=device, dtype=dtype))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, a_dim, device=device, dtype=dtype))

        self.scale_shift_table_a2v_ca_audio = nn.Parameter(
            torch.empty(5, a_dim, device=device, dtype=dtype)
        )
        self.scale_shift_table_a2v_ca_video = nn.Parameter(
            torch.empty(5, v_dim, device=device, dtype=dtype)
        )

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice = slice(None, None)
    ):
        if isinstance(timestep, CompressedTimestep):
            return timestep.expand_for_computation(scale_shift_table, batch_size, indices)

        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ):
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
        )

        return (*scale_shift_ada_values, *gate_ada_values)

    def _apply_text_cross_attention(
        self, x, context, attn, scale_shift_table, prompt_scale_shift_table,
        timestep, prompt_timestep, attention_mask, transformer_options,
    ):
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(
                scale_shift_table, x.shape[0], timestep, slice(6, 9)
            )
            return apply_cross_attention_adaln(
                x, context, attn, shift_q, scale_q, gate,
                prompt_scale_shift_table, prompt_timestep,
                attention_mask, transformer_options,
            )
        return attn(
            rms_norm(x), context=context,
            mask=attention_mask, transformer_options=transformer_options,
        )

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor], v_context=None, a_context=None, attention_mask=None, v_timestep=None, a_timestep=None,
        v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None, v_cross_scale_shift_timestep=None, a_cross_scale_shift_timestep=None,
        v_cross_gate_timestep=None, a_cross_gate_timestep=None, transformer_options=None, self_attention_mask=None,
        v_prompt_timestep=None, a_prompt_timestep=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if transformer_options is None:
            transformer_options = {}
        run_vx = transformer_options.get("run_vx", True)
        run_ax = transformer_options.get("run_ax", True)

        vx, ax = x
        run_ax = run_ax and ax.numel() > 0
        run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0
        run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True)

        if run_vx:
            vshift_msa, vscale_msa = (self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(0, 2)))
            norm_vx = rms_norm(vx) * (1 + vscale_msa) + vshift_msa
            del vshift_msa, vscale_msa
            attn1_out = self.attn1(norm_vx, pe=v_pe, mask=self_attention_mask, transformer_options=transformer_options)
            del norm_vx
            vgate_msa = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(2, 3))[0]
            vx.addcmul_(attn1_out, vgate_msa)
            del vgate_msa, attn1_out
            vx.add_(self._apply_text_cross_attention(
                vx, v_context, self.attn2, self.scale_shift_table,
                getattr(self, 'prompt_scale_shift_table', None),
                v_timestep, v_prompt_timestep, attention_mask, transformer_options,)
            )

        if run_ax:
            ashift_msa, ascale_msa = (self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(0, 2)))
            norm_ax = rms_norm(ax) * (1 + ascale_msa) + ashift_msa
            del ashift_msa, ascale_msa
            attn1_out = self.audio_attn1(norm_ax, pe=a_pe, transformer_options=transformer_options)
            del norm_ax
            agate_msa = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(2, 3))[0]
            ax.addcmul_(attn1_out, agate_msa)
            del agate_msa, attn1_out
            ax.add_(self._apply_text_cross_attention(
                ax, a_context, self.audio_attn2, self.audio_scale_shift_table,
                getattr(self, 'audio_prompt_scale_shift_table', None),
                a_timestep, a_prompt_timestep, attention_mask, transformer_options,)
            )

        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx)
            ax_norm3 = rms_norm(ax)

            if run_a2v:
                scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[:2]
                scale_ca_video_hidden_states_a2v_v, shift_ca_video_hidden_states_a2v_v = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[:2]

                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v_v) + shift_ca_video_hidden_states_a2v_v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
                del scale_ca_video_hidden_states_a2v_v, shift_ca_video_hidden_states_a2v_v, scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v

                a2v_out = self.audio_to_video_attn(vx_scaled, context=ax_scaled, pe=v_cross_pe, k_pe=a_cross_pe, transformer_options=transformer_options)
                del vx_scaled, ax_scaled

                gate_out_a2v = self.get_ada_values(self.scale_shift_table_a2v_ca_video[4:, :], vx.shape[0], v_cross_gate_timestep)[0]
                vx.addcmul_(a2v_out, gate_out_a2v)
                del gate_out_a2v, a2v_out

            if run_v2a:
                scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[2:4]
                scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[2:4]

                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
                del scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a, scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a

                v2a_out = self.video_to_audio_attn(ax_scaled, context=vx_scaled, pe=a_cross_pe, k_pe=v_cross_pe, transformer_options=transformer_options)
                del ax_scaled, vx_scaled

                gate_out_v2a = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[4:, :], ax.shape[0], a_cross_gate_timestep)[0]
                ax.addcmul_(v2a_out, gate_out_v2a)
                del gate_out_v2a, v2a_out

            del vx_norm3, ax_norm3

        if run_vx:
            vshift_mlp, vscale_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(3, 5))
            vx_scaled = rms_norm(vx) * (1 + vscale_mlp) + vshift_mlp
            del vshift_mlp, vscale_mlp

            ff_out = self.ff(vx_scaled)
            del vx_scaled

            vgate_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(5, 6))[0]
            vx.addcmul_(ff_out, vgate_mlp)
            del vgate_mlp, ff_out

        if run_ax:
            ashift_mlp, ascale_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 5))
            ax_scaled = rms_norm(ax) * (1 + ascale_mlp) + ashift_mlp
            del ashift_mlp, ascale_mlp

            ff_out = self.audio_ff(ax_scaled)
            del ax_scaled

            agate_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(5, 6))[0]
            ax.addcmul_(ff_out, agate_mlp)
            del agate_mlp, ff_out

        return vx, ax


class LTXAVModel(LTXVModel):
    def __init__(
        self,
        in_channels=128,
        audio_in_channels=128,
        cross_attention_dim=4096,
        audio_cross_attention_dim=2048,
        attention_head_dim=128,
        audio_attention_head_dim=64,
        num_attention_heads=32,
        audio_num_attention_heads=32,
        caption_channels=3840,
        num_layers=48,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        audio_positional_embedding_max_pos=[20],
        causal_temporal_positioning=False,
        vae_scale_factors=(8, 32, 32),
        use_middle_indices_grid=False,
        timestep_scale_multiplier=1000.0,
        av_ca_timestep_scale_multiplier=1.0,
        apply_gated_attention=False,
        caption_proj_before_connector=False,
        cross_attention_adaln=False,
        dtype=None,
        device=None,
        **kwargs,
    ):
        self.audio_in_channels = audio_in_channels
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.audio_attention_head_dim = audio_attention_head_dim
        self.audio_num_attention_heads = audio_num_attention_heads
        self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
        self.apply_gated_attention = apply_gated_attention

        self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
        self.audio_out_channels = audio_in_channels

        self.num_audio_channels = 8
        self.audio_frequency_bins = 16

        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

        super().__init__(
            in_channels=in_channels,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            caption_channels=caption_channels,
            num_layers=num_layers,
            positional_embedding_theta=positional_embedding_theta,
            positional_embedding_max_pos=positional_embedding_max_pos,
            causal_temporal_positioning=causal_temporal_positioning,
            vae_scale_factors=vae_scale_factors,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            caption_proj_before_connector=caption_proj_before_connector,
            cross_attention_adaln=cross_attention_adaln,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    def _init_model_components(self, device, dtype, **kwargs):
        self.audio_patchify_proj = nn.Linear(
            self.audio_in_channels, self.audio_inner_dim, bias=True, dtype=dtype, device=device
        )

        audio_embedding_coefficient = ADALN_CROSS_ATTN_PARAMS_COUNT if self.cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=audio_embedding_coefficient,
            use_additional_conditions=False,
            dtype=dtype,
            device=device,
        )

        if self.cross_attention_adaln:
            self.audio_prompt_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim,
                embedding_coefficient=2,
                use_additional_conditions=False,
                dtype=dtype,
                device=device,
            )
        else:
            self.audio_prompt_adaln_single = None

        num_scale_shift_values = 4
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            use_additional_conditions=False,
            embedding_coefficient=num_scale_shift_values,
            dtype=dtype,
            device=device,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            use_additional_conditions=False,
            embedding_coefficient=1,
            dtype=dtype,
            device=device,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            use_additional_conditions=False,
            embedding_coefficient=num_scale_shift_values,
            dtype=dtype,
            device=device,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            use_additional_conditions=False,
            embedding_coefficient=1,
            dtype=dtype,
            device=device,
        )

        if self.caption_proj_before_connector:
            if self.caption_projection_first_linear:
                self.audio_caption_projection = NormSingleLinearTextProjection(
                    in_features=self.caption_channels,
                    hidden_size=self.audio_inner_dim,
                    dtype=dtype,
                    device=device,
                )
            else:
                self.audio_caption_projection = lambda a: a
        else:
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels,
                hidden_size=self.audio_inner_dim,
                dtype=dtype,
                device=device,
            )

        connector_split_rope = kwargs.get("rope_type", "split") == "split"
        connector_gated_attention = kwargs.get("connector_apply_gated_attention", False)
        attention_head_dim = kwargs.get("connector_attention_head_dim", 128)
        num_attention_heads = kwargs.get("connector_num_attention_heads", 30)
        num_layers = kwargs.get("connector_num_layers", 2)

        self.audio_embeddings_connector = Embeddings1DConnector(
            attention_head_dim=kwargs.get("audio_connector_attention_head_dim", attention_head_dim),
            num_attention_heads=kwargs.get("audio_connector_num_attention_heads", num_attention_heads),
            num_layers=num_layers,
            split_rope=connector_split_rope,
            double_precision_rope=True,
            apply_gated_attention=connector_gated_attention,
            dtype=dtype,
            device=device,
        )

        self.video_embeddings_connector = Embeddings1DConnector(
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            split_rope=connector_split_rope,
            double_precision_rope=True,
            apply_gated_attention=connector_gated_attention,
            dtype=dtype,
            device=device,
        )

    def preprocess_text_embeds(self, context, unprocessed=False):
        if not unprocessed:
            if context.shape[-1] in (self.cross_attention_dim + self.audio_cross_attention_dim, self.caption_channels * 2):
                return context
        if context.shape[-1] == self.cross_attention_dim + self.audio_cross_attention_dim:
            context_vid = context[:, :, :self.cross_attention_dim]
            context_audio = context[:, :, self.cross_attention_dim:]
        else:
            context_vid = context
            context_audio = context
        if self.caption_proj_before_connector:
            context_vid = self.caption_projection(context_vid)
            context_audio = self.audio_caption_projection(context_audio)
        out_vid = self.video_embeddings_connector(context_vid)[0]
        out_audio = self.audio_embeddings_connector(context_audio)[0]
        return torch.concat((out_vid, out_audio), dim=-1)

    def _init_transformer_blocks(self, device, dtype, **kwargs):
        self.transformer_blocks = nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    v_dim=self.inner_dim,
                    a_dim=self.audio_inner_dim,
                    v_heads=self.num_attention_heads,
                    a_heads=self.audio_num_attention_heads,
                    vd_head=self.attention_head_dim,
                    ad_head=self.audio_attention_head_dim,
                    v_context_dim=self.cross_attention_dim,
                    a_context_dim=self.audio_cross_attention_dim,
                    apply_gated_attention=self.apply_gated_attention,
                    cross_attention_adaln=self.cross_attention_adaln,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _init_output_components(self, device, dtype):
        super()._init_output_components(device, dtype)
        self.audio_scale_shift_table = nn.Parameter(
            torch.empty(2, self.audio_inner_dim, dtype=dtype, device=device)
        )
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.audio_proj_out = nn.Linear(
            self.audio_inner_dim, self.audio_out_channels, dtype=dtype, device=device
        )
        self.a_patchifier = AudioPatchifier(1, start_end=True)

    def separate_audio_and_video_latents(self, x, audio_length):
        vx = x[0]
        ax = x[1] if len(x) > 1 else torch.zeros(
            (vx.shape[0], self.num_audio_channels, 0, self.audio_frequency_bins),
            device=vx.device, dtype=vx.dtype
        )
        return vx, ax

    def recombine_audio_and_video_latents(self, vx, ax, target_shape=None):
        if ax.numel() == 0:
            return vx
        else:
            return [vx, ax]

    def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        audio_length = kwargs.get("audio_length", 0)
        vx, ax = self.separate_audio_and_video_latents(x, audio_length)

        has_spatial_mask = False
        if denoise_mask is not None:
            for frame_idx in range(denoise_mask.shape[2]):
                frame_mask = denoise_mask[0, 0, frame_idx]
                if frame_mask.numel() > 0 and frame_mask.min() != frame_mask.max():
                    has_spatial_mask = True
                    break

        [vx, v_pixel_coords, additional_args] = super()._process_input(
            vx, keyframe_idxs, denoise_mask, **kwargs
        )
        additional_args["has_spatial_mask"] = has_spatial_mask

        ax, a_latent_coords = self.a_patchifier.patchify(ax)

        # -- ID-LoRA: prepend reference audio tokens with negative positions --
        ref_audio = kwargs.get("ref_audio")
        ref_audio_seq_len = 0
        if ref_audio is not None:
            ref_patchifier = self.a_patchifier.copy_with_shift(0)
            ref_tokens, _ = ref_patchifier.patchify(ref_audio)
            ref_audio_seq_len = ref_tokens.shape[1]
            B = ax.shape[0]

            # Compute negative temporal coordinates (ComfyUI ID-LoRA convention):
            # use positive indices then offset so reference ends just before t=0.
            p = self.a_patchifier
            tpl = p.hop_length * p.audio_latent_downsample_factor / p.sample_rate
            ref_start = p._get_audio_latent_time_in_sec(
                0, ref_audio_seq_len, torch.float32, ax.device
            )
            ref_end = p._get_audio_latent_time_in_sec(
                1, ref_audio_seq_len + 1, torch.float32, ax.device
            )
            time_offset = ref_end[-1].item() + tpl
            ref_start = (ref_start - time_offset).unsqueeze(0).expand(B, -1).unsqueeze(1)
            ref_end = (ref_end - time_offset).unsqueeze(0).expand(B, -1).unsqueeze(1)
            ref_coords = torch.stack([ref_start, ref_end], dim=-1)

            additional_args["ref_audio_seq_len"] = ref_audio_seq_len
            additional_args["target_audio_seq_len"] = ax.shape[1]
            ax = torch.cat([ref_tokens, ax], dim=1)
            a_latent_coords = torch.cat([ref_coords.to(a_latent_coords), a_latent_coords], dim=2)

        if ref_audio is None:
            additional_args["ref_audio_seq_len"] = 0
            additional_args["target_audio_seq_len"] = ax.shape[1]

        ax = self.audio_patchify_proj(ax)

        return [vx, ax], [v_pixel_coords, a_latent_coords], additional_args

    def _prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        grid_mask = kwargs.get("grid_mask", None)
        if grid_mask is not None:
            timestep = timestep[:, grid_mask]

        timestep_scaled = timestep * self.timestep_scale_multiplier

        v_timestep, v_embedded_timestep = self.adaln_single(
            timestep_scaled.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )

        orig_shape = kwargs.get("orig_shape")
        has_spatial_mask = kwargs.get("has_spatial_mask", None)
        v_patches_per_frame = None
        if not has_spatial_mask and orig_shape is not None and len(orig_shape) == 5:
            v_patches_per_frame = orig_shape[3] * orig_shape[4]

        v_timestep = CompressedTimestep(v_timestep.view(batch_size, -1, v_timestep.shape[-1]), v_patches_per_frame)
        v_embedded_timestep = CompressedTimestep(v_embedded_timestep.view(batch_size, -1, v_embedded_timestep.shape[-1]), v_patches_per_frame)

        v_prompt_timestep = compute_prompt_timestep(
            self.prompt_adaln_single, timestep_scaled, batch_size, hidden_dtype
        )

        a_timestep = kwargs.get("a_timestep")
        ref_audio_seq_len = kwargs.get("ref_audio_seq_len", 0)

        # Prepend zero timesteps for ID-LoRA reference tokens
        if a_timestep is not None and ref_audio_seq_len > 0:
            ref_zeros = torch.zeros(
                a_timestep.shape[0], ref_audio_seq_len,
                device=a_timestep.device, dtype=a_timestep.dtype,
            )
            if a_timestep.ndim == 1:
                a_timestep = a_timestep.unsqueeze(1).expand(-1, kwargs.get("target_audio_seq_len", 1))
            a_timestep = torch.cat([ref_zeros, a_timestep], dim=1)

        if a_timestep is not None:
            a_timestep_scaled = a_timestep * self.timestep_scale_multiplier
            a_timestep_flat = a_timestep_scaled.flatten()
            timestep_flat = timestep_scaled.flatten()
            av_ca_factor = self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier

            av_ca_audio_scale_shift_timestep, _ = self.av_ca_audio_scale_shift_adaln_single(
                timestep.max().expand_as(a_timestep_flat),
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            av_ca_video_scale_shift_timestep, _ = self.av_ca_video_scale_shift_adaln_single(
                a_timestep.max().expand_as(timestep_flat),
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            av_ca_a2v_gate_noise_timestep, _ = self.av_ca_a2v_gate_adaln_single(
                a_timestep.max().expand_as(timestep_flat) * av_ca_factor,
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            av_ca_v2a_gate_noise_timestep, _ = self.av_ca_v2a_gate_adaln_single(
                timestep.max().expand_as(a_timestep_flat) * av_ca_factor,
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )

            cross_av_timestep_ss = [
                av_ca_audio_scale_shift_timestep.view(batch_size, -1, av_ca_audio_scale_shift_timestep.shape[-1]),
                CompressedTimestep(av_ca_video_scale_shift_timestep.view(batch_size, -1, av_ca_video_scale_shift_timestep.shape[-1]), v_patches_per_frame),
                CompressedTimestep(av_ca_a2v_gate_noise_timestep.view(batch_size, -1, av_ca_a2v_gate_noise_timestep.shape[-1]), v_patches_per_frame),
                av_ca_v2a_gate_noise_timestep.view(batch_size, -1, av_ca_v2a_gate_noise_timestep.shape[-1]),
            ]

            a_timestep_emb, a_embedded_timestep = self.audio_adaln_single(
                a_timestep_flat,
                {"resolution": None, "aspect_ratio": None},
                batch_size=batch_size,
                hidden_dtype=hidden_dtype,
            )
            a_timestep_emb = a_timestep_emb.view(batch_size, -1, a_timestep_emb.shape[-1])
            a_embedded_timestep = a_embedded_timestep.view(batch_size, -1, a_embedded_timestep.shape[-1])

            a_prompt_timestep = compute_prompt_timestep(
                self.audio_prompt_adaln_single, a_timestep_scaled, batch_size, hidden_dtype
            )
        else:
            a_timestep_emb = timestep_scaled
            a_embedded_timestep = kwargs.get("embedded_timestep")
            cross_av_timestep_ss = []
            a_prompt_timestep = None

        return [v_timestep, a_timestep_emb, cross_av_timestep_ss, v_prompt_timestep, a_prompt_timestep], [
            v_embedded_timestep,
            a_embedded_timestep,
        ], None

    def _prepare_context(self, context, batch_size, x, attention_mask=None):
        vx = x[0]
        ax = x[1]
        video_dim = vx.shape[-1]
        audio_dim = ax.shape[-1]

        v_context_dim = self.caption_channels if self.caption_proj_before_connector is False else video_dim
        a_context_dim = self.caption_channels if self.caption_proj_before_connector is False else audio_dim

        v_context, a_context = torch.split(
            context, [v_context_dim, a_context_dim], len(context.shape) - 1
        )

        v_context, attention_mask = super()._prepare_context(
            v_context, batch_size, vx, attention_mask
        )
        if self.caption_proj_before_connector is False:
            a_context = self.audio_caption_projection(a_context)
        a_context = a_context.view(batch_size, -1, audio_dim)

        return [v_context, a_context], attention_mask

    def _prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
        v_pixel_coords = pixel_coords[0]
        v_pe = super()._prepare_positional_embeddings(v_pixel_coords, frame_rate, x_dtype)

        a_latent_coords = pixel_coords[1]
        a_pe = self._precompute_freqs_cis(
            a_latent_coords,
            dim=self.audio_inner_dim,
            out_dtype=x_dtype,
            max_pos=self.audio_positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.audio_num_attention_heads,
        )

        max_pos = max(
            self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0]
        )
        v_pixel_coords = v_pixel_coords.to(torch.float32)
        v_pixel_coords[:, 0] = v_pixel_coords[:, 0] * (1.0 / frame_rate)
        av_cross_video_freq_cis = self._precompute_freqs_cis(
            v_pixel_coords[:, 0:1, :],
            dim=self.audio_cross_attention_dim,
            out_dtype=x_dtype,
            max_pos=[max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.audio_num_attention_heads,
        )
        av_cross_audio_freq_cis = self._precompute_freqs_cis(
            a_latent_coords[:, 0:1, :],
            dim=self.audio_cross_attention_dim,
            out_dtype=x_dtype,
            max_pos=[max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.audio_num_attention_heads,
        )

        return [(v_pe, av_cross_video_freq_cis), (a_pe, av_cross_audio_freq_cis)]

    def _process_transformer_blocks(
        self, x, context, attention_mask, timestep, pe, transformer_options={}, self_attention_mask=None, **kwargs
    ):
        vx = x[0]
        ax = x[1]
        v_context = context[0]
        a_context = context[1]
        v_timestep = timestep[0]
        a_timestep = timestep[1]
        v_pe, av_cross_video_freq_cis = pe[0]
        a_pe, av_cross_audio_freq_cis = pe[1]

        (
            av_ca_audio_scale_shift_timestep,
            av_ca_video_scale_shift_timestep,
            av_ca_a2v_gate_noise_timestep,
            av_ca_v2a_gate_noise_timestep,
        ) = timestep[2]

        v_prompt_timestep = timestep[3]
        a_prompt_timestep = timestep[4]

        for block in self.transformer_blocks:
            vx, ax = block(
                (vx, ax),
                v_context=v_context,
                a_context=a_context,
                attention_mask=attention_mask,
                v_timestep=v_timestep,
                a_timestep=a_timestep,
                v_pe=v_pe,
                a_pe=a_pe,
                v_cross_pe=av_cross_video_freq_cis,
                a_cross_pe=av_cross_audio_freq_cis,
                v_cross_scale_shift_timestep=av_ca_video_scale_shift_timestep,
                a_cross_scale_shift_timestep=av_ca_audio_scale_shift_timestep,
                v_cross_gate_timestep=av_ca_a2v_gate_noise_timestep,
                a_cross_gate_timestep=av_ca_v2a_gate_noise_timestep,
                transformer_options=transformer_options,
                self_attention_mask=self_attention_mask,
                v_prompt_timestep=v_prompt_timestep,
                a_prompt_timestep=a_prompt_timestep,
            )

        return [vx, ax]

    def _process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        vx = x[0]
        ax = x[1]
        v_embedded_timestep = embedded_timestep[0]
        a_embedded_timestep = embedded_timestep[1]

        # Trim ID-LoRA reference tokens before unpatchify
        ref_n = kwargs.get("ref_audio_seq_len", 0)
        if ref_n > 0:
            ax = ax[:, ref_n:]
            a_embedded_timestep = a_embedded_timestep[:, ref_n:]

        if isinstance(v_embedded_timestep, CompressedTimestep):
            v_embedded_timestep = v_embedded_timestep.expand()

        vx = super()._process_output(vx, v_embedded_timestep, keyframe_idxs, **kwargs)

        a_scale_shift_values = (
            self.audio_scale_shift_table[None, None].to(device=a_embedded_timestep.device, dtype=a_embedded_timestep.dtype)
            + a_embedded_timestep[:, :, None]
        )
        a_shift, a_scale = a_scale_shift_values[:, :, 0], a_scale_shift_values[:, :, 1]

        ax = self.audio_norm_out(ax)
        ax = ax * (1 + a_scale) + a_shift
        ax = self.audio_proj_out(ax)

        ax = self.a_patchifier.unpatchify(
            ax, channels=self.num_audio_channels, freq=self.audio_frequency_bins
        )

        original_shape = kwargs.get("av_orig_shape")
        return self.recombine_audio_and_video_latents(vx, ax, original_shape)

    def forward(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        frame_rate=25,
        transformer_options={},
        keyframe_idxs=None,
        **kwargs,
    ):
        if isinstance(timestep, (tuple, list)) and len(timestep) == 2:
            v_timestep, a_timestep = timestep
            kwargs["a_timestep"] = a_timestep
            timestep = v_timestep
        else:
            kwargs["a_timestep"] = timestep

        return super().forward(
            x,
            timestep,
            context,
            attention_mask,
            frame_rate,
            transformer_options,
            keyframe_idxs,
            **kwargs,
        )
