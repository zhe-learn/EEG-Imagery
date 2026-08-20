"""
Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N., Cavigelli, L., & Benini, L. (2006).
EEG-TCNet: An accurate temporal convolutional network for embedded motor-imagery brain-machine interfaces 2020.
2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Toronto, ON, Canada, 2020, pp. 2958-2965.

"""

import torch
from einops.layers.torch import Rearrange
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


class CausalConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, inp):
        return super().forward(nnf.pad(inp, (self.__padding, 0)))


class _TCNBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.project_channels = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.project_channels = nn.Identity()
        self.final_nonlinearity = nn.ReLU()

    def forward(self, inp):
        residual = self.project_channels(inp)
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.drop2(out)
        return self.final_nonlinearity(out + residual)


class EEGTCNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 22,
            n_classes: int = 4,
            n_samples: int = 1000,
            layers: int = 2,
            tcn_kernel_size: int = 4,
            tcn_filters: int = 12,
            tcn_dropout: float = 0.3,
            f1: int = 8,
            depth_multiplier: int = 2,
            kernel_length: int = 32,
            eeg_dropout: float = 0.2
    ):
        super().__init__()
        f2 = f1 * depth_multiplier

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(f1, momentum=0.01, eps=0.001),
            Conv2dWithConstraint(f1, f2, (in_channels, 1), bias=False, groups=f1, max_norm=1),
            nn.BatchNorm2d(f2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(eeg_dropout),
            nn.Conv2d(f2, f2, (1, 16), padding="same", groups=f2, bias=False),
            nn.Conv2d(f2, f2, 1, bias=False),
            nn.BatchNorm2d(f2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(eeg_dropout),
            Rearrange("b c 1 t -> b c t")
        )

        tcn_inputs = [f2] + (layers - 1) * [tcn_filters]
        dilations = [2 ** idx for idx in range(layers)]
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(
                in_channels=in_ch,
                out_channels=tcn_filters,
                kernel_size=tcn_kernel_size,
                dilation=dilation,
                dropout=tcn_dropout
            ) for in_ch, dilation in zip(tcn_inputs, dilations)
        ])

        self.classifier = LinearWithConstraint(tcn_filters, n_classes, max_norm=0.25)

    def forward(self, inp):
        inp = self.eegnet(inp)
        for blk in self.tcn_blocks:
            inp = blk(inp)
        inp = self.classifier(inp[:, :, -1])
        return inp


if __name__ == '__main__':
    from thop import profile

    model = EEGTCNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
