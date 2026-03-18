"""Video VAE for LTX 2.3.

Ported from ComfyUI's comfy/ldm/lightricks/vae/causal_video_autoencoder.py
with all comfy.* dependencies removed.
"""

from __future__ import annotations

import math
import threading
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from .conv_nd_factory import make_conv_nd, make_linear_nd
from .causal_conv3d import CausalConv3d
from .pixel_norm import PixelNorm


# ---------------------------------------------------------------------------
# Helpers ported from comfy.ldm.lightricks.model
# ---------------------------------------------------------------------------

def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1
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


class _Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        return _get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class _TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, dtype=None, device=None, **kwargs):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, dtype=dtype, device=device)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device)

    def forward(self, sample, condition=None):
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = _Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = _TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype, device=device,
        )

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))


# ---------------------------------------------------------------------------
# Trivial helper from comfy.ldm.modules.diffusionmodules.model
# ---------------------------------------------------------------------------

def _torch_cat_if_needed(xl, dim):
    xl = [x for x in xl if x is not None and x.shape[dim] > 0]
    if len(xl) > 1:
        return torch.cat(xl, dim)
    elif len(xl) == 1:
        return xl[0]
    else:
        return None


# ---------------------------------------------------------------------------
# Video VAE memory scaling
# ---------------------------------------------------------------------------

MIN_VRAM_FOR_CHUNK_SCALING = 6 * 1024 ** 3
MAX_VRAM_FOR_CHUNK_SCALING = 24 * 1024 ** 3
MIN_CHUNK_SIZE = 32 * 1024 ** 2
MAX_CHUNK_SIZE = 128 * 1024 ** 2


def _get_total_gpu_memory(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory
    return MAX_VRAM_FOR_CHUNK_SCALING


def _get_max_chunk_size(device: torch.device) -> int:
    total_memory = _get_total_gpu_memory(device)
    if total_memory <= MIN_VRAM_FOR_CHUNK_SCALING:
        return MIN_CHUNK_SIZE
    if total_memory >= MAX_VRAM_FOR_CHUNK_SCALING:
        return MAX_CHUNK_SIZE
    interp = (total_memory - MIN_VRAM_FOR_CHUNK_SCALING) / (
        MAX_VRAM_FOR_CHUNK_SCALING - MIN_VRAM_FOR_CHUNK_SCALING
    )
    return int(MIN_CHUNK_SIZE + interp * (MAX_CHUNK_SIZE - MIN_CHUNK_SIZE))


# ---------------------------------------------------------------------------
# Meta-device detection
# ---------------------------------------------------------------------------

def _in_meta_context():
    return torch.device("meta") == torch.empty(0).device


# ---------------------------------------------------------------------------
# CausalConv3d helpers
# ---------------------------------------------------------------------------

def _mark_conv3d_ended(module):
    tid = threading.get_ident()
    for _, m in module.named_modules():
        if isinstance(m, CausalConv3d):
            current = m.temporal_cache_state.get(tid, (None, False))
            m.temporal_cache_state[tid] = (current[0], True)


def _split2(tensor, split_point, dim=2):
    return torch.split(tensor, [split_point, tensor.shape[dim] - split_point], dim=dim)


def _add_exchange_cache(dest, cache_in, new_input, dim=2):
    if dest is not None:
        if cache_in is not None:
            cache_to_dest = min(dest.shape[dim], cache_in.shape[dim])
            lead_in_dest, dest = _split2(dest, cache_to_dest, dim=dim)
            lead_in_source, cache_in = _split2(cache_in, cache_to_dest, dim=dim)
            lead_in_dest.add_(lead_in_source)
        body, new_input = _split2(new_input, dest.shape[dim], dim)
        dest.add_(body)
    return _torch_cat_if_needed([cache_in, new_input], dim=dim)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        dims: Union[int, Tuple[int, int]] = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        blocks: List[Tuple[str, int | dict]] = [("res_x", 1)],
        base_channels: int = 128,
        norm_num_groups: int = 32,
        patch_size: Union[int, Tuple[int]] = 1,
        norm_layer: str = "group_norm",
        latent_log_var: str = "per_channel",
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self.blocks_desc = blocks

        in_channels = in_channels * patch_size**2
        output_channel = base_channels

        self.conv_in = make_conv_nd(
            dims=dims, in_channels=in_channels, out_channels=output_channel,
            kernel_size=3, stride=1, padding=1, causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in blocks:
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {"num_layers": block_params}

            if block_name == "res_x":
                block = UNetMidBlock3D(
                    dims=dims, in_channels=input_channel,
                    num_layers=block_params["num_layers"],
                    resnet_eps=1e-6, resnet_groups=norm_num_groups,
                    norm_layer=norm_layer, spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "res_x_y":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = ResnetBlock3D(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    eps=1e-6, groups=norm_num_groups, norm_layer=norm_layer,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time":
                block = make_conv_nd(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    kernel_size=3, stride=(2, 1, 1), causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space":
                block = make_conv_nd(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    kernel_size=3, stride=(1, 2, 2), causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all":
                block = make_conv_nd(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    kernel_size=3, stride=(2, 2, 2), causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all_x_y":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = make_conv_nd(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    kernel_size=3, stride=(2, 2, 2), causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    stride=(2, 2, 2), spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    stride=(1, 2, 2), spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    stride=(2, 1, 1), spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"unknown block: {block_name}")

            self.down_blocks.append(block)

        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(
                num_channels=output_channel, num_groups=norm_num_groups, eps=1e-6
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        elif norm_layer == "layer_norm":
            self.conv_norm_out = LayerNorm(output_channel, eps=1e-6)

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == "per_channel":
            conv_out_channels *= 2
        elif latent_log_var == "uniform":
            conv_out_channels += 1
        elif latent_log_var == "constant":
            conv_out_channels += 1
        elif latent_log_var != "none":
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")
        self.conv_out = make_conv_nd(
            dims, output_channel, conv_out_channels, 3, padding=1,
            causal=True, spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False

    def forward_orig(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        sample = _patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        for down_block in self.down_blocks:
            sample = checkpoint_fn(down_block)(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == "uniform":
            last_channel = sample[:, -1:, ...]
            num_dims = sample.dim()
            if num_dims == 4:
                repeated_last_channel = last_channel.repeat(1, sample.shape[1] - 2, 1, 1)
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            elif num_dims == 5:
                repeated_last_channel = last_channel.repeat(1, sample.shape[1] - 2, 1, 1, 1)
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            else:
                raise ValueError(f"Invalid input shape: {sample.shape}")
        elif self.latent_log_var == "constant":
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        return sample

    def forward(self, *args, **kwargs):
        _mark_conv3d_ended(self)
        try:
            return self.forward_orig(*args, **kwargs)
        finally:
            tid = threading.get_ident()
            for _, module in self.named_modules():
                if hasattr(module, "temporal_cache_state"):
                    module.temporal_cache_state.pop(tid, None)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        in_channels: int = 3,
        out_channels: int = 3,
        blocks: List[Tuple[str, int | dict]] = [("res_x", 1)],
        base_channels: int = 128,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        patch_size: int = 1,
        norm_layer: str = "group_norm",
        causal: bool = True,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.layers_per_block = layers_per_block
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.blocks_desc = blocks

        output_channel = base_channels
        for block_name, block_params in list(reversed(blocks)):
            block_params = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                output_channel = output_channel * block_params.get("multiplier", 2)
            if block_name == "compress_all":
                output_channel = output_channel * block_params.get("multiplier", 1)
            if block_name == "compress_space":
                output_channel = output_channel * block_params.get("multiplier", 1)
            if block_name == "compress_time":
                output_channel = output_channel * block_params.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims, in_channels, output_channel, kernel_size=3, stride=1, padding=1,
            causal=True, spatial_padding_mode=spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(blocks)):
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {"num_layers": block_params}

            if block_name == "res_x":
                block = UNetMidBlock3D(
                    dims=dims, in_channels=input_channel,
                    num_layers=block_params["num_layers"],
                    resnet_eps=1e-6, resnet_groups=norm_num_groups,
                    norm_layer=norm_layer,
                    inject_noise=block_params.get("inject_noise", False),
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "res_x_y":
                output_channel = output_channel // block_params.get("multiplier", 2)
                block = ResnetBlock3D(
                    dims=dims, in_channels=input_channel, out_channels=output_channel,
                    eps=1e-6, groups=norm_num_groups, norm_layer=norm_layer,
                    inject_noise=block_params.get("inject_noise", False),
                    timestep_conditioning=False,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time":
                output_channel = output_channel // block_params.get("multiplier", 1)
                block = DepthToSpaceUpsample(
                    dims=dims, in_channels=input_channel, stride=(2, 1, 1),
                    out_channels_reduction_factor=block_params.get("multiplier", 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space":
                output_channel = output_channel // block_params.get("multiplier", 1)
                block = DepthToSpaceUpsample(
                    dims=dims, in_channels=input_channel, stride=(1, 2, 2),
                    out_channels_reduction_factor=block_params.get("multiplier", 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all":
                output_channel = output_channel // block_params.get("multiplier", 1)
                block = DepthToSpaceUpsample(
                    dims=dims, in_channels=input_channel, stride=(2, 2, 2),
                    residual=block_params.get("residual", False),
                    out_channels_reduction_factor=block_params.get("multiplier", 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"unknown layer: {block_name}")

            self.up_blocks.append(block)

        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(
                num_channels=output_channel, num_groups=norm_num_groups, eps=1e-6
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        elif norm_layer == "layer_norm":
            self.conv_norm_out = LayerNorm(output_channel, eps=1e-6)

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims, output_channel, out_channels, 3, padding=1,
            causal=True, spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False
        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(
                torch.tensor(1000.0, dtype=torch.float32)
            )
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                output_channel * 2, 0,
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, output_channel))
        else:
            self.register_buffer(
                "last_scale_shift_table",
                torch.tensor(
                    [0.0, 0.0],
                    device="cpu" if _in_meta_context() else None
                ).unsqueeze(1).expand(2, output_channel),
                persistent=False,
            )

    def forward_orig(
        self,
        sample: torch.FloatTensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size = sample.shape[0]

        _mark_conv3d_ended(self.conv_in)
        sample = self.conv_in(sample, causal=self.causal)

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        timestep_shift_scale = None
        if self.timestep_conditioning:
            assert timestep is not None, "should pass timestep with timestep_conditioning=True"
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(dtype=sample.dtype, device=sample.device)
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                resolution=None, aspect_ratio=None,
                batch_size=sample.shape[0], hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(
                batch_size, embedded_timestep.shape[-1], 1, 1, 1
            )
            ada_values = self.last_scale_shift_table[
                None, ..., None, None, None
            ].to(device=sample.device, dtype=sample.dtype) + embedded_timestep.reshape(
                batch_size, 2, -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            timestep_shift_scale = ada_values.unbind(dim=1)

        output = []
        max_chunk_size = _get_max_chunk_size(sample.device)

        # Store intermediate_device as CPU to avoid GPU OOM during chunked decode
        intermediate_device = torch.device("cpu")

        def run_up(idx, sample_ref, ended):
            sample = sample_ref[0]
            sample_ref[0] = None
            if idx >= len(self.up_blocks):
                sample = self.conv_norm_out(sample)
                if timestep_shift_scale is not None:
                    shift, scale = timestep_shift_scale
                    sample = sample * (1 + scale) + shift
                sample = self.conv_act(sample)
                if ended:
                    _mark_conv3d_ended(self.conv_out)
                sample = self.conv_out(sample, causal=self.causal)
                if sample is not None and sample.shape[2] > 0:
                    output.append(sample.to(intermediate_device))
                return

            up_block = self.up_blocks[idx]
            if ended:
                _mark_conv3d_ended(up_block)
            if self.timestep_conditioning and isinstance(up_block, UNetMidBlock3D):
                sample = checkpoint_fn(up_block)(
                    sample, causal=self.causal, timestep=scaled_timestep
                )
            else:
                sample = checkpoint_fn(up_block)(sample, causal=self.causal)

            if sample is None or sample.shape[2] == 0:
                return

            total_bytes = sample.numel() * sample.element_size()
            num_chunks = (total_bytes + max_chunk_size - 1) // max_chunk_size

            if num_chunks == 1:
                next_sample_ref = [sample]
                del sample
                run_up(idx + 1, next_sample_ref, ended)
                return
            else:
                samples = torch.chunk(sample, chunks=num_chunks, dim=2)
                for chunk_idx, sample1 in enumerate(samples):
                    run_up(idx + 1, [sample1], ended and chunk_idx == len(samples) - 1)

        run_up(0, [sample], True)
        sample = torch.cat(output, dim=2)

        sample = _unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample

    def forward(self, *args, **kwargs):
        try:
            return self.forward_orig(*args, **kwargs)
        finally:
            for _, module in self.named_modules():
                tid = threading.get_ident()
                if hasattr(module, "temporal_cache_state"):
                    module.temporal_cache_state.pop(tid, None)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: str = "group_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                in_channels * 4, 0,
            )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock3D(
                    dims=dims, in_channels=in_channels, out_channels=in_channels,
                    eps=resnet_eps, groups=resnet_groups, dropout=dropout,
                    norm_layer=norm_layer, inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        timestep_embed = None
        if self.timestep_conditioning:
            assert timestep is not None
            batch_size = hidden_states.shape[0]
            timestep_embed = self.time_embedder(
                timestep=timestep.flatten(), resolution=None,
                aspect_ratio=None, batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            timestep_embed = timestep_embed.view(
                batch_size, timestep_embed.shape[-1], 1, 1, 1
            )

        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states, causal=causal, timestep=timestep_embed)

        return hidden_states


class SpaceToDepthDownsample(nn.Module):
    def __init__(self, dims, in_channels, out_channels, stride, spatial_padding_mode):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv_nd(
            dims=dims, in_channels=in_channels,
            out_channels=out_channels // math.prod(stride),
            kernel_size=3, stride=1, causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, x, causal: bool = True):
        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)

        x_in = rearrange(
            x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0], p2=self.stride[1], p3=self.stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
        x_in = x_in.mean(dim=2)

        x = self.conv(x, causal=causal)
        x = rearrange(
            x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0], p2=self.stride[1], p3=self.stride[2],
        )

        x = x + x_in
        return x


class DepthToSpaceUpsample(nn.Module):
    def __init__(
        self, dims, in_channels, stride, residual=False,
        out_channels_reduction_factor=1, spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = (
            math.prod(stride) * in_channels // out_channels_reduction_factor
        )
        self.conv = make_conv_nd(
            dims=dims, in_channels=in_channels, out_channels=self.out_channels,
            kernel_size=3, stride=1, causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        self.temporal_cache_state = {}

    def forward(self, x, causal: bool = True, timestep: Optional[torch.Tensor] = None):
        tid = threading.get_ident()
        cached, drop_first_conv, drop_first_res = self.temporal_cache_state.get(tid, (None, True, True))
        y = self.conv(x, causal=causal)
        y = rearrange(
            y, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=self.stride[0], p2=self.stride[1], p3=self.stride[2],
        )
        if self.stride[0] == 2 and y.shape[2] > 0 and drop_first_conv:
            y = y[:, :, 1:, :, :]
            drop_first_conv = False
        if self.residual:
            x_in = rearrange(
                x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.stride[0], p2=self.stride[1], p3=self.stride[2],
            )
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2 and x_in.shape[2] > 0 and drop_first_res:
                x_in = x_in[:, :, 1:, :, :]
                drop_first_res = False

            if y.shape[2] == 0:
                y = None

            cached = _add_exchange_cache(y, cached, x_in, dim=2)
            self.temporal_cache_state[tid] = (cached, drop_first_conv, drop_first_res)

        else:
            self.temporal_cache_state[tid] = (None, drop_first_conv, False)

        return y


class LayerNorm(nn.Module):
    def __init__(self, dim, eps, elementwise_affine=True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: str = "group_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        if norm_layer == "group_norm":
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        elif norm_layer == "pixel_norm":
            self.norm1 = PixelNorm()
        elif norm_layer == "layer_norm":
            self.norm1 = LayerNorm(in_channels, eps=eps, elementwise_affine=True)

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            causal=True, spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        if norm_layer == "group_norm":
            self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        elif norm_layer == "pixel_norm":
            self.norm2 = PixelNorm()
        elif norm_layer == "layer_norm":
            self.norm2 = LayerNorm(out_channels, eps=eps, elementwise_affine=True)

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims, out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            causal=True, spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        self.conv_shortcut = (
            make_linear_nd(dims=dims, in_channels=in_channels, out_channels=out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.norm3 = (
            LayerNorm(in_channels, eps=eps, elementwise_affine=True)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(
                torch.randn(4, in_channels) / in_channels**0.5
            )
        else:
            self.register_buffer(
                "scale_shift_table",
                torch.tensor(
                    [0.0, 0.0, 0.0, 0.0],
                    device="cpu" if _in_meta_context() else None
                ).unsqueeze(1).expand(4, in_channels),
                persistent=False,
            )

        self.temporal_cache_state = {}

    def _feed_spatial_noise(
        self, hidden_states: torch.FloatTensor, per_channel_scale: torch.FloatTensor
    ) -> torch.FloatTensor:
        spatial_shape = hidden_states.shape[-2:]
        device = hidden_states.device
        dtype = hidden_states.dtype
        spatial_noise = torch.randn(spatial_shape, device=device, dtype=dtype)[None]
        scaled_noise = (spatial_noise * per_channel_scale)[None, :, None, ...]
        hidden_states = hidden_states + scaled_noise
        return hidden_states

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]

        hidden_states = self.norm1(hidden_states)
        if self.timestep_conditioning:
            assert timestep is not None
            ada_values = self.scale_shift_table[
                None, ..., None, None, None
            ].to(device=hidden_states.device, dtype=hidden_states.dtype) + timestep.reshape(
                batch_size, 4, -1,
                timestep.shape[-3], timestep.shape[-2], timestep.shape[-1],
            )
            shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale1) + shift1

        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states, self.per_channel_scale1.to(device=hidden_states.device, dtype=hidden_states.dtype)
            )

        hidden_states = self.norm2(hidden_states)

        if self.timestep_conditioning:
            hidden_states = hidden_states * (1 + scale2) + shift2

        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states, self.per_channel_scale2.to(device=hidden_states.device, dtype=hidden_states.dtype)
            )

        input_tensor = self.norm3(input_tensor)
        input_tensor = self.conv_shortcut(input_tensor)

        tid = threading.get_ident()
        cached = self.temporal_cache_state.get(tid, None)
        cached = _add_exchange_cache(hidden_states, cached, input_tensor, dim=2)
        self.temporal_cache_state[tid] = cached

        return hidden_states


# ---------------------------------------------------------------------------
# Patchify / unpatchify
# ---------------------------------------------------------------------------

def _patchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x, "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t, q=patch_size_hw, r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def _unpatchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x, "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t, q=patch_size_hw, r=patch_size_hw,
        )
    return x


# ---------------------------------------------------------------------------
# Per-channel statistics processor
# ---------------------------------------------------------------------------

class _Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(128))
        self.register_buffer("mean-of-means", torch.empty(128))

    def un_normalize(self, x):
        return (x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)) + self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)

    def normalize(self, x):
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)


# ---------------------------------------------------------------------------
# Top-level VideoVAE
# ---------------------------------------------------------------------------

class VideoVAE(nn.Module):
    def __init__(self, version=0, config=None):
        super().__init__()

        if config is None:
            config = self._get_default_config(version)

        self.config = config
        self.timestep_conditioning = config.get("timestep_conditioning", False)
        self.decode_noise_scale = config.get("decode_noise_scale", 0.025)
        self.decode_timestep = config.get("decode_timestep", 0.05)
        double_z = config.get("double_z", True)
        latent_log_var = config.get(
            "latent_log_var", "per_channel" if double_z else "none"
        )

        self.encoder = Encoder(
            dims=config["dims"],
            in_channels=config.get("in_channels", 3),
            out_channels=config["latent_channels"],
            blocks=config.get("encoder_blocks", config.get("blocks")),
            patch_size=config.get("patch_size", 1),
            latent_log_var=latent_log_var,
            norm_layer=config.get("norm_layer", "group_norm"),
            spatial_padding_mode=config.get("spatial_padding_mode", "zeros"),
            base_channels=config.get("encoder_base_channels", 128),
        )

        self.decoder = Decoder(
            dims=config["dims"],
            in_channels=config["latent_channels"],
            out_channels=config.get("out_channels", 3),
            blocks=config.get("decoder_blocks", config.get("blocks")),
            base_channels=config.get("decoder_base_channels", 128),
            patch_size=config.get("patch_size", 1),
            norm_layer=config.get("norm_layer", "group_norm"),
            causal=config.get("causal_decoder", False),
            timestep_conditioning=self.timestep_conditioning,
            spatial_padding_mode=config.get("spatial_padding_mode", "reflect"),
        )

        self.per_channel_statistics = _Processor()

    def encode(self, x):
        frames_count = x.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                "(e.g., 1, 9, 17, ...). Please check your input."
            )
        means, logvar = torch.chunk(self.encoder(x), 2, dim=1)
        return self.per_channel_statistics.normalize(means)

    def decode(self, x):
        if self.timestep_conditioning:
            x = torch.randn_like(x) * self.decode_noise_scale + (1.0 - self.decode_noise_scale) * x
        return self.decoder(self.per_channel_statistics.un_normalize(x), timestep=self.decode_timestep)

    @staticmethod
    def _get_default_config(version):
        if version == 0:
            return {
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3, "in_channels": 3, "out_channels": 3,
                "latent_channels": 128,
                "blocks": [
                    ["res_x", 4], ["compress_all", 1], ["res_x_y", 1],
                    ["res_x", 3], ["compress_all", 1], ["res_x_y", 1],
                    ["res_x", 3], ["compress_all", 1], ["res_x", 3], ["res_x", 4],
                ],
                "scaling_factor": 1.0, "norm_layer": "pixel_norm",
                "patch_size": 4, "latent_log_var": "uniform",
                "use_quant_conv": False, "causal_decoder": False,
            }
        elif version == 1:
            return {
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3, "in_channels": 3, "out_channels": 3,
                "latent_channels": 128,
                "decoder_blocks": [
                    ["res_x", {"num_layers": 5, "inject_noise": True}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 6, "inject_noise": True}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 7, "inject_noise": True}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 8, "inject_noise": False}],
                ],
                "encoder_blocks": [
                    ["res_x", {"num_layers": 4}], ["compress_all", {}],
                    ["res_x_y", 1], ["res_x", {"num_layers": 3}],
                    ["compress_all", {}], ["res_x_y", 1],
                    ["res_x", {"num_layers": 3}], ["compress_all", {}],
                    ["res_x", {"num_layers": 3}], ["res_x", {"num_layers": 4}],
                ],
                "scaling_factor": 1.0, "norm_layer": "pixel_norm",
                "patch_size": 4, "latent_log_var": "uniform",
                "use_quant_conv": False, "causal_decoder": False,
                "timestep_conditioning": True,
            }
        else:
            return {
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3, "in_channels": 3, "out_channels": 3,
                "latent_channels": 128,
                "encoder_blocks": [
                    ["res_x", {"num_layers": 4}],
                    ["compress_space_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 6}],
                    ["compress_time_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 6}],
                    ["compress_all_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 2}],
                    ["compress_all_res", {"multiplier": 2}],
                    ["res_x", {"num_layers": 2}],
                ],
                "decoder_blocks": [
                    ["res_x", {"num_layers": 5, "inject_noise": False}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 5, "inject_noise": False}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 5, "inject_noise": False}],
                    ["compress_all", {"residual": True, "multiplier": 2}],
                    ["res_x", {"num_layers": 5, "inject_noise": False}],
                ],
                "scaling_factor": 1.0, "norm_layer": "pixel_norm",
                "patch_size": 4, "latent_log_var": "uniform",
                "use_quant_conv": False, "causal_decoder": False,
                "timestep_conditioning": True,
            }
