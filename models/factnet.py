"""
Ke, S., Yang, batch_size., Qin, Y., Rong, nnf., Zhang, J., & Zheng, Y. (2024).
FACT-Net: a frequency adapter CNN with temporal-periodicity inception for fast and accurate MI-EEG decoding.
IEEE Transactions on Neural Systems and Rehabilitation Engineering, 32, 4131-4142.

"""

import math

import numpy as np
import torch
import torch.fft
from torch import nn
from torch.nn import functional as nnf


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, inp):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=float(self.max_norm)
            )
        return super().forward(inp)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, inp):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=float(self.max_norm)
            )
        return super().forward(inp)


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = np.random.choice(seq_len // 2, modes, replace=False)
    elif mode_select_method == 'segmented_random':
        segments = np.array_split(np.arange(seq_len // 2), modes)
        index = [np.random.choice(segment) for segment in segments]
    else:
        index = np.arange(modes)
    index.sort()
    return index


class FA(nn.Module):
    def __init__(self, seq_len, modes=64):
        super().__init__()
        self.radio = 1
        self.index = get_frequency_modes(seq_len)
        self.fweights_size = [min(modes, math.ceil(len(self.index) / self.radio)) + 1, 1]
        self.fweights = nn.Parameter(torch.zeros(self.fweights_size), requires_grad=True)
        self.fweights_im = nn.Parameter(torch.zeros(self.fweights_size), requires_grad=True)
        self.dropout = nn.Dropout(p=0.3)

    def compl_mul1d(self, input, weights, idx, flag='freq'):
        if flag == 'freq':
            rate = 0
            weight = weights[idx].unsqueeze(-1).unsqueeze(-1)
            if idx == 0:
                all_weight = weight
            elif idx == 1:
                all_weight = weight + rate * weights[idx + 1].unsqueeze(-1).unsqueeze(-1)
            elif idx == self.fweights_size[0] - 1:
                all_weight = weight + rate * weights[idx - 1].unsqueeze(-1).unsqueeze(-1)
            else:
                all_weight = weight + rate * weights[idx + 1].unsqueeze(-1).unsqueeze(-1)
            return input * all_weight
        return None

    def forward(self, inp):
        inp = inp.squeeze(dim=1)
        batch_size, feat_dim, seq_len = inp.shape
        fft_dim = -1
        fft_signal = torch.fft.rfftn(inp, dim=fft_dim, norm='ortho')
        fft_re = fft_signal.real
        fft_im = fft_signal.imag
        out_re = torch.zeros(batch_size, feat_dim, seq_len // 2 + 1, device=inp.device)
        out_im = torch.zeros(batch_size, feat_dim, seq_len // 2 + 1, device=inp.device)
        for w_idx, idx in enumerate(self.index):
            if w_idx == 0:
                out_re[:, :, w_idx] = self.compl_mul1d(fft_re[:, :, idx], self.fweights, 0, flag='freq')
                out_im[:, :, w_idx] = self.compl_mul1d(fft_im[:, :, idx], self.fweights_im, 0, flag='freq')
            else:
                out_re[:, :, w_idx] = self.compl_mul1d(
                    fft_re[:, :, idx], self.fweights, int(w_idx / self.radio) + 1, flag='freq'
                )
                out_im[:, :, w_idx] = self.compl_mul1d(
                    fft_im[:, :, idx], self.fweights_im, int(w_idx / self.radio) + 1, flag='freq'
                )
        self.fweights.data = torch.renorm(
            self.fweights.data, p=2, dim=0, maxnorm=1)

        self.fweights_im.data = torch.renorm(
            self.fweights.data, p=2, dim=0, maxnorm=1)

        fft_out = torch.complex(fft_re + out_re, fft_im + out_im)
        inp = torch.fft.irfftn(fft_out, s=seq_len, dim=2, norm='ortho')
        if len(inp.shape) != 4:
            inp = torch.unsqueeze(inp, 1)
        return inp


class EEGDepthAware(nn.Module):
    def __init__(self, width, channels, kernel_size):
        super().__init__()
        self.channels = channels
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, width))
        self.conv = nn.Conv2d(1, 1, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, inp):
        pooled = self.adaptive_pool(inp)
        transposed = pooled.transpose(-2, -3)
        gated = self.conv(transposed)
        gated = self.softmax(gated)
        gated = gated.transpose(-2, -3)
        return gated * self.channels * inp


class FFT_Based_Refactor(nn.Module):
    def __init__(self, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.top_list = None

    def forward(self, inp):
        if self.top_list is None or 1:
            xf = torch.fft.rfft(inp, dim=1)
            frequency_list = abs(xf).mean(0).mean(-1)
            frequency_list[0] = 0
            value, top_list = torch.topk(frequency_list, self.top_k)
            top_list = top_list.detach().cpu().numpy()
            length = top_list.shape[0]
            for idx in range(length - 1, 0, -1):
                top_list[idx] = top_list[idx - 1]
            top_list[0] = 1
            self.top_list = top_list
        else:
            xf = torch.fft.rfft(inp, dim=1)
        period = inp.shape[1] // self.top_list
        return period, abs(xf).mean(-1)[:, self.top_list]


class Multi_periodicity_Inception(nn.Module):
    def __init__(self, in_channels, out_channels, n_kernels, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_kernels = n_kernels

        kernels = []
        for idx in range(self.n_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * idx + 1, padding=idx, groups=in_channels))
        kernels.append(nn.AvgPool2d(kernel_size=(3, 3), padding=(1, 1)))

        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, inp):
        res_list = []
        for idx in range(self.n_kernels):
            res_list.append(self.kernels[idx](inp))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TPI(nn.Module):
    def __init__(self, seq_len, d_model, n_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = 3
        self.fft_get_p = FFT_Based_Refactor(self.top_k)
        self.d_model = d_model
        self.conv = nn.Sequential(
            Multi_periodicity_Inception(self.d_model, self.d_model, n_kernels=n_kernels),
            nn.GELU(),
            Multi_periodicity_Inception(self.d_model, self.d_model, n_kernels=n_kernels)
        )

    def forward(self, inp):
        batch_size, tokens, n_channels = inp.size()
        period_list, period_weight = self.fft_get_p(inp)
        res = []
        for idx in range(self.top_k):
            period = period_list[idx]
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([inp.shape[0], (length - self.seq_len), inp.shape[2]]).to(inp.device)
                out = torch.cat([inp, padding], dim=1)
            else:
                length = self.seq_len
                out = inp
            out = out.reshape(batch_size, length // period, period, n_channels).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, n_channels)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = nnf.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, tokens, n_channels, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + inp
        return res


class FACTNet(nn.Module):
    def __init__(
            self,
            in_channels=22,
            n_classes=4,
            n_samples=1000,
            f1=8,
            depth_multiplier=2,
            f2=16,
            use_fa=True
    ):
        super().__init__()

        self.use_fa = use_fa
        if self.use_fa:
            self.fa = FA(seq_len=n_samples)

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, 64),
                stride=1,
                padding='same',
                bias=False,
                groups=1
            ),
            nn.BatchNorm2d(num_features=f1)
        )

        self.channel_conv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=f1,
                out_channels=f1 * depth_multiplier,
                kernel_size=(in_channels, 1),
                groups=f1,
                bias=False,
                max_norm=1.
            ),
            nn.BatchNorm2d(num_features=f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8)
            ),
            nn.Dropout(p=0.3)
        )

        self.depth_separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f1 * depth_multiplier,
                kernel_size=(1, 16),
                stride=1,
                padding='same',
                groups=f1 * depth_multiplier,
                bias=False
            ),
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f2,
                kernel_size=(1, 1),
                groups=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=f2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8)
            ),
            nn.Dropout(p=0.3)
        )

        self.depth_aware_conv = EEGDepthAware(width=n_samples // 64, channels=f2, kernel_size=7)
        self.layer = 1
        n_kernels = 4
        self.layer_norm = nn.LayerNorm(f2)
        self.model = nn.ModuleList([TPI(n_samples // 64, f2, n_kernels) for _ in range(self.layer)])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                in_features=f2 * (n_samples // 64),
                out_features=n_classes,
                max_norm=.25
            ),
            nn.Softmax(dim=-1),
        )

    def forward(self, inp):
        if len(inp.shape) != 4:
            inp = torch.unsqueeze(inp, 1)

        if self.use_fa:
            inp = self.fa(inp)

        inp = self.temporal_conv(inp)
        inp = self.channel_conv(inp)
        inp = self.depth_separable_conv(inp)

        inp = self.depth_aware_conv(inp)

        inp = torch.squeeze(inp, dim=2)
        inp = inp.permute(0, 2, 1)
        for idx in range(self.layer):
            inp = self.layer_norm(self.model[idx](inp))

        inp = self.classifier(inp)

        return inp


if __name__ == '__main__':
    from thop import profile

    model = FACTNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
