"""
Altaheri, H., Muhammad, G., & Alsulaiman, M. (2022).
Physics-informed attention temporal convolutional network for EEG-based motor imagery classification.
IEEE transactions on industrial informatics, 19(2), 2249-2258.

"""

import numpy as np
import torch
from einops import rearrange
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


class _ConvBlock(nn.Module):
    def __init__(
            self,
            f1: int = 16,
            kernel_size: int = 64,
            pool_size: int = 8,
            depth_multiplier: int = 2,
            in_channels: int = 22,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.temporal_conv = nn.Conv2d(
            1, f1, (1, kernel_size), padding=(0, kernel_size // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(f1, momentum=0.01, eps=0.001)

        self.spat_conv = Conv2dWithConstraint(
            f1, f1 * depth_multiplier, (in_channels, 1), bias=False, groups=f1, max_norm=1.0
        )
        self.bn2 = nn.BatchNorm2d(f1 * depth_multiplier, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool_size))
        self.drop1 = nn.Dropout(dropout)

        self.conv = nn.Conv2d(f1 * depth_multiplier, f1 * depth_multiplier, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(f1 * depth_multiplier, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 7))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, inp):
        inp = self.temporal_conv(inp)
        inp = self.bn1(inp)

        inp = self.spat_conv(inp)
        inp = self.bn2(inp)
        inp = self.nonlinearity1(inp)
        inp = self.pool1(inp)
        inp = self.drop1(inp)

        inp = self.conv(inp)
        inp = self.bn3(inp)
        inp = self.nonlinearity2(inp)
        inp = self.pool2(inp)
        inp = self.drop2(inp)

        return inp


class _AttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            key_dim=8,
            n_heads=2,
            dropout=0.5
    ):
        super().__init__()
        self.n_heads = n_heads

        self.w_qs = nn.Linear(d_model, n_heads * key_dim)
        self.w_ks = nn.Linear(d_model, n_heads * key_dim)
        self.w_vs = nn.Linear(d_model, n_heads * key_dim)

        self.fc = nn.Linear(n_heads * key_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp):
        residual = inp
        inp = self.layer_norm(inp)
        queries = rearrange(self.w_qs(inp), "b l (head keys) -> head b l keys", head=self.n_heads)
        keys = rearrange(self.w_ks(inp), "b t (head keys) -> head b t keys", head=self.n_heads)
        values = rearrange(self.w_vs(inp), "b t (head values) -> head b t values", head=self.n_heads)
        attention_weights = torch.einsum("hblk, hbtk -> hblt", [queries, keys]) / np.sqrt(queries.shape[-1])
        attention_weights = torch.softmax(attention_weights, dim=3)

        out = torch.einsum("hblt,hbtv->hblv", [attention_weights, values])
        out = rearrange(out, "head b l values -> b l (head values)")
        out = self.dropout(self.fc(out))
        out = out + residual

        return out


class TCNBlock(nn.Module):
    def __init__(
            self,
            kernel_size: int = 4,
            filters: int = 32,
            dilation: int = 1,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            filters, filters, kernel_size=kernel_size, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(filters, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            filters, filters, kernel_size=kernel_size, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(filters, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.nonlinearity3 = nn.ELU()

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, inp):
        inp = self.drop1(self.nonlinearity1(self.bn1(self.conv1(inp))))
        inp = self.drop2(self.nonlinearity2(self.bn2(self.conv2(inp))))
        inp = self.nonlinearity3(inp + inp)
        return inp


class TCN(nn.Module):
    def __init__(
            self,
            depth: int = 2,
            kernel_size: int = 4,
            filters: int = 32,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for idx in range(depth):
            dilation = 2 ** idx
            self.blocks.append(TCNBlock(kernel_size, filters, dilation, dropout))

    def forward(self, inp):
        for block in self.blocks:
            inp = block(inp)
        return inp


class ATCBlock(nn.Module):
    def __init__(
            self,
            d_model: int = 32,
            key_dim: int = 8,
            n_heads: int = 2,
            dropout_attn: float = 0.3,
            tcn_depth: int = 2,
            tcn_kernel_size: int = 4,
            dropout_tcn: float = 0.3,
            n_classes: int = 4,
    ):
        super().__init__()
        self.attention_block = _AttentionBlock(d_model, key_dim, n_heads, dropout_attn)
        self.rearrange = Rearrange("b seq c -> b c seq")
        self.tcn = TCN(tcn_depth, tcn_kernel_size, d_model, dropout_tcn)
        self.linear = LinearWithConstraint(d_model, n_classes, max_norm=0.25)

    def forward(self, inp):
        inp = self.attention_block(inp)
        inp = self.rearrange(inp)
        inp = self.tcn(inp)
        inp = self.linear(inp[:, :, -1])
        return inp


class ATCNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 22,
            n_classes: int = 4,
            n_samples: int = 1000,
            f1: int = 16,
            conv_kernel_size: int = 64,
            pool_size: int = 8,
            depth_multiplier: int = 2,
            dropout_conv: float = 0.3,
            d_model: int = 32,
            key_dim: int = 8,
            n_heads: int = 2,
            dropout_attn: float = 0.5,
            tcn_depth: int = 2,
            tcn_kernel_size: int = 4,
            dropout_tcn: float = 0.3,
            n_windows: int = 5,
    ):
        super().__init__()
        self.conv_block = _ConvBlock(
            f1, conv_kernel_size, pool_size, depth_multiplier, in_channels, dropout_conv
        )
        self.rearrange = Rearrange("b c 1 seq -> b seq c")

        self.atc_blocks = nn.ModuleList(
            [
                ATCBlock(
                    d_model,
                    key_dim,
                    n_heads,
                    dropout_attn,
                    tcn_depth,
                    tcn_kernel_size,
                    dropout_tcn,
                    n_classes,
                )
                for _ in range(n_windows)
            ]
        )
        self.n_windows = n_windows
        self.n_classes = n_classes

    def forward(self, inp):
        inp = self.conv_block(inp)
        inp = self.rearrange(inp)

        batch_size, seq_len, _ = inp.shape
        block_output = torch.zeros(batch_size, self.n_classes, dtype=inp.dtype, device=inp.device)
        for idx, block in enumerate(self.atc_blocks):
            block_output = block_output + block(
                inp[:, idx: (seq_len - self.n_windows + idx + 1), :]
            )

        block_output = block_output / self.n_windows

        return block_output


if __name__ == '__main__':
    from thop import profile

    model = ATCNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
