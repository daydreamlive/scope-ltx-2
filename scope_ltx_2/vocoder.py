"""Standalone vocoder for LTX 2.3 audio decode.

Adapted from ComfyUI's comfy/ldm/lightricks/vocoders/vocoder.py to work
without comfy.* dependencies. Supports HiFi-GAN (resblock "1"/"2") and
BigVGAN v2 (resblock "AMP1") with bandwidth extension (BWE).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.1


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def _sinc(x: torch.Tensor):
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def _kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, padding=True,
                 padding_mode="replicate", kernel_size=12):
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", _kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        filt = self.filter.expand(C, -1, -1).to(dtype=x.dtype, device=x.device)
        return F.conv1d(x, filt, stride=self.stride, groups=C)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, persistent=True, window_type="kaiser"):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            t = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            t_clamped = t.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = torch.cos(t_clamped * math.pi / lowpass_filter_width / 2) ** 2
            filt = (torch.sinc(t) * window * rolloff / ratio).view(1, 1, -1)
        else:
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            filt = _kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size)

        self.register_buffer("filter", filt, persistent=persistent)

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        filt = self.filter.expand(C, -1, -1).to(dtype=x.dtype, device=x.device)
        x = self.ratio * F.conv_transpose1d(x, filt, stride=self.stride, groups=C)
        return x[..., self.pad_left:-self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio,
            kernel_size=int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2,
                 up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha_logscale=True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        a = self.alpha.unsqueeze(0).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        b = self.beta.unsqueeze(0).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        if self.alpha_logscale:
            a = torch.exp(a)
            b = torch.exp(b)
        return x + (1.0 / (b + 1e-9)) * torch.sin(x * a).pow(2)


class Snake(nn.Module):
    def __init__(self, in_features, alpha_logscale=True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        a = self.alpha.unsqueeze(0).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        if self.alpha_logscale:
            a = torch.exp(a)
        return x + (1.0 / (a + 1e-9)) * torch.sin(x * a).pow(2)


class AMPBlock1(nn.Module):
    """BigVGAN v2 anti-aliased multi-periodicity residual block."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation="snake"):
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      dilation=d, padding=_get_padding(kernel_size, d))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      dilation=1, padding=_get_padding(kernel_size, 1))
            for _ in dilation
        ])
        self.acts1 = nn.ModuleList([Activation1d(act_cls(channels)) for _ in dilation])
        self.acts2 = nn.ModuleList([Activation1d(act_cls(channels)) for _ in dilation])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
        return x


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      dilation=d, padding=_get_padding(kernel_size, d))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      dilation=1, padding=_get_padding(kernel_size, 1))
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class Vocoder(nn.Module):
    """HiFi-GAN / BigVGAN v2 vocoder."""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        resblock_kernel_sizes = config.get("resblock_kernel_sizes", [3, 7, 11])
        upsample_rates = config.get("upsample_rates", [5, 4, 2, 2, 2])
        upsample_kernel_sizes = config.get("upsample_kernel_sizes", [16, 16, 8, 4, 4])
        resblock_dilation_sizes = config.get("resblock_dilation_sizes", [[1, 3, 5]] * 3)
        upsample_initial_channel = config.get("upsample_initial_channel", 1024)
        stereo = config.get("stereo", True)
        activation = config.get("activation", "snake")
        use_bias_at_final = config.get("use_bias_at_final", True)

        self.output_sample_rate = config.get("output_sample_rate")
        self.resblock = config.get("resblock", "1")
        self.use_tanh_at_final = config.get("use_tanh_at_final", True)
        self.apply_final_activation = config.get("apply_final_activation", True)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        if self.resblock == "AMP1":
            resblock_cls = AMPBlock1
        elif self.resblock == "1":
            resblock_cls = ResBlock1
        else:
            resblock_cls = ResBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** i),
                upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2,
            ))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                if self.resblock == "AMP1":
                    self.resblocks.append(resblock_cls(ch, k, d, activation=activation))
                else:
                    self.resblocks.append(resblock_cls(ch, k, d))

        out_channels = 2 if stereo else 1
        ch = upsample_initial_channel // (2 ** self.num_upsamples)
        if self.resblock == "AMP1":
            act_cls = SnakeBeta if activation == "snakebeta" else Snake
            self.act_post = Activation1d(act_cls(ch))
        else:
            self.act_post = nn.LeakyReLU()

        self.conv_post = nn.Conv1d(ch, out_channels, 7, 1, padding=3, bias=use_bias_at_final)
        self.upsample_factor = int(np.prod([self.ups[i].stride[0] for i in range(len(self.ups))]))

    def forward(self, x):
        if x.dim() == 4:
            assert x.shape[1] == 2
            x = torch.cat((x[:, 0, :, :], x[:, 1, :, :]), dim=1)
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            if self.resblock != "AMP1":
                x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.act_post(x)
        x = self.conv_post(x)

        if self.apply_final_activation:
            if self.use_tanh_at_final:
                x = torch.tanh(x)
            else:
                x = torch.clamp(x, -1, 1)
        return x


class _STFTFn(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length))

    def forward(self, y: torch.Tensor):
        if y.dim() == 2:
            y = y.unsqueeze(1)
        left_pad = max(0, self.win_length - self.hop_length)
        y = F.pad(y, (left_pad, 0))
        fwd = self.forward_basis.to(dtype=y.dtype, device=y.device)
        spec = F.conv1d(y, fwd, stride=self.hop_length, padding=0)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag.float(), real.float()).to(real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    def __init__(self, filter_length, hop_length, win_length, n_mel_channels,
                 sampling_rate, mel_fmin, mel_fmax):
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(n_mel_channels, n_freqs))

    def mel_spectrogram(self, y):
        magnitude, phase = self.stft_fn(y)
        energy = torch.norm(magnitude, dim=1)
        mel_basis = self.mel_basis.to(dtype=magnitude.dtype, device=y.device)
        mel = torch.matmul(mel_basis, magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class VocoderWithBWE(nn.Module):
    """Vocoder with bandwidth extension for higher sample rate output."""

    def __init__(self, config):
        super().__init__()
        vocoder_config = config["vocoder"]
        bwe_config = config["bwe"]

        self.vocoder = Vocoder(config=vocoder_config)
        self.bwe_generator = Vocoder(config={**bwe_config, "apply_final_activation": False})

        self.input_sample_rate = bwe_config["input_sampling_rate"]
        self.output_sample_rate = bwe_config["output_sampling_rate"]
        self.hop_length = bwe_config["hop_length"]

        self.mel_stft = MelSTFT(
            filter_length=bwe_config["n_fft"],
            hop_length=bwe_config["hop_length"],
            win_length=bwe_config["n_fft"],
            n_mel_channels=bwe_config["num_mels"],
            sampling_rate=bwe_config["input_sampling_rate"],
            mel_fmin=0.0,
            mel_fmax=bwe_config["input_sampling_rate"] / 2.0,
        )
        self.resampler = UpSample1d(
            ratio=bwe_config["output_sampling_rate"] // bwe_config["input_sampling_rate"],
            persistent=False,
            window_type="hann",
        )

    def _compute_mel(self, audio):
        B, C, T = audio.shape
        flat = audio.reshape(B * C, -1)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
        return mel.reshape(B, C, mel.shape[1], mel.shape[2])

    def forward(self, mel_spec):
        x = self.vocoder(mel_spec)
        _, _, T_low = x.shape
        T_out = T_low * self.output_sample_rate // self.input_sample_rate

        remainder = T_low % self.hop_length
        if remainder != 0:
            x = F.pad(x, (0, self.hop_length - remainder))

        mel = self._compute_mel(x)
        residual = self.bwe_generator(mel)
        skip = self.resampler(x)
        return torch.clamp(residual + skip, -1, 1)[..., :T_out]
