import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DualConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(DualConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if kernel_size == (1, 1, 1):
            raise ValueError(
                "kernel_size must be greater than 1. Use make_linear_nd instead."
            )
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.groups = groups
        self.bias = bias

        intermediate_channels = (
            out_channels if in_channels < out_channels else in_channels
        )

        self.weight1 = nn.Parameter(
            torch.Tensor(
                intermediate_channels,
                in_channels // groups,
                1,
                kernel_size[1],
                kernel_size[2],
            )
        )
        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(intermediate_channels))
        else:
            self.register_parameter("bias1", None)

        self.weight2 = nn.Parameter(
            torch.Tensor(
                out_channels, intermediate_channels // groups, kernel_size[0], 1, 1
            )
        )
        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)
        if bias:
            self.bias2 = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias2", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1)
            nn.init.uniform_(self.bias1, -bound1, bound1)
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2)
            nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, x, use_conv3d=False, skip_time_conv=False):
        if use_conv3d:
            return self.forward_with_3d(x=x, skip_time_conv=skip_time_conv)
        else:
            return self.forward_with_2d(x=x, skip_time_conv=skip_time_conv)

    def forward_with_3d(self, x, skip_time_conv):
        x = F.conv3d(
            x, self.weight1, self.bias1, self.stride1, self.padding1,
            self.dilation1, self.groups, padding_mode=self.padding_mode,
        )
        if skip_time_conv:
            return x
        x = F.conv3d(
            x, self.weight2, self.bias2, self.stride2, self.padding2,
            self.dilation2, self.groups, padding_mode=self.padding_mode,
        )
        return x

    def forward_with_2d(self, x, skip_time_conv):
        b, c, d, h, w = x.shape
        x = rearrange(x, "b c d h w -> (b d) c h w")
        weight1 = self.weight1.squeeze(2)
        stride1 = (self.stride1[1], self.stride1[2])
        padding1 = (self.padding1[1], self.padding1[2])
        dilation1 = (self.dilation1[1], self.dilation1[2])
        x = F.conv2d(
            x, weight1, self.bias1, stride1, padding1, dilation1,
            self.groups, padding_mode=self.padding_mode,
        )
        _, _, h, w = x.shape
        if skip_time_conv:
            x = rearrange(x, "(b d) c h w -> b c d h w", b=b)
            return x
        x = rearrange(x, "(b d) c h w -> (b h w) c d", b=b)
        weight2 = self.weight2.squeeze(-1).squeeze(-1)
        stride2 = self.stride2[0]
        padding2 = self.padding2[0]
        dilation2 = self.dilation2[0]
        x = F.conv1d(
            x, weight2, self.bias2, stride2, padding2, dilation2,
            self.groups, padding_mode=self.padding_mode,
        )
        x = rearrange(x, "(b h w) c d -> b c d h w", b=b, h=h, w=w)
        return x

    @property
    def weight(self):
        return self.weight2
