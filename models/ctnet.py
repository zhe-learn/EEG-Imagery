"""
Zhao, W., Jiang, X., Zhang, B., Xiao, S., & Weng, S. (2024).
CTNet: a convolutional transformer network for EEG-based motor imagery classification.
Scientific reports, 14(1), 20237.

"""

import math

import torch
import torch.nn.functional as nnf
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.keys = nn.Linear(d_model, d_model)
        self.queries = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, inp: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(inp), "b n (h d) -> b h n d", h=self.n_heads)
        keys = rearrange(self.keys(inp), "b n (h d) -> b h n d", h=self.n_heads)
        values = rearrange(self.values(inp), "b n (h d) -> b h n d", h=self.n_heads)
        attention_scores = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            attention_scores.mask_fill(~mask, fill_value)

        scaling = self.d_model ** (1 / 2)
        attention_probs = nnf.softmax(attention_scores / scaling, dim=-1)
        attention_probs = self.att_drop(attention_probs)
        out = torch.einsum('bhal, bhlv -> bhav ', attention_probs, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, d_model, expansion, dropout):
        super().__init__(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features, n_classes)
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, d_model, dropout):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inp, **kwargs):
        residual = inp
        out = self.fn(inp, **kwargs)
        out = self.layernorm(self.drop(out) + residual)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, d_model, n_heads, dropout=0.5, expansion=4, ff_dropout=0.5):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    MultiHeadAttention(d_model, n_heads, dropout),
                ), d_model, dropout
            ),
            ResidualAdd(
                nn.Sequential(
                    FeedForwardBlock(d_model, expansion=expansion, dropout=ff_dropout),
                ), d_model, dropout
            )
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, n_heads, depth, d_model):
        super().__init__(*[TransformerEncoderBlock(d_model, n_heads) for _ in range(depth)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, d_model))

    def forward(self, inp):
        inp = inp + self.encoding[:, :inp.shape[1], :]
        return self.dropout(inp)


class PatchEmbeddingCNN(nn.Module):
    def __init__(
            self,
            f1,
            kernel_size,
            depth_multiplier,
            pooling_size1,
            pooling_size2,
            dropout,
            in_channels,
    ):
        super().__init__()
        f2 = depth_multiplier * f1
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (in_channels, 1), (1, 1), groups=f1, padding='valid', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout),
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout)
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, inp: Tensor) -> Tensor:
        batch_size, _, _, _ = inp.shape
        inp = self.cnn_module(inp)
        inp = self.projection(inp)
        return inp


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(
            self,
            in_channels,
            f1,
            kernel_size,
            depth_multiplier,
            pooling_size1,
            pooling_size2,
            dropout,
    ):
        super().__init__(
            PatchEmbeddingCNN(
                f1=f1, kernel_size=kernel_size, depth_multiplier=depth_multiplier, pooling_size1=pooling_size1,
                pooling_size2=pooling_size2, dropout=dropout, in_channels=in_channels
            ),
        )


class CTNet(nn.Module):
    def __init__(
            self,
            in_channels=22,
            n_classes=4,
            n_samples=1000,
            n_heads=4,
            d_model=40,
            depth=6,
            f1=20,
            kernel_size=64,
            depth_multiplier=2,
            pooling_size1=8,
            pooling_size2=8,
            dropout=0.3,
    ):
        super().__init__()
        self.number_class, self.in_channels, self.n_samples = n_classes, in_channels, n_samples
        self.d_model = d_model

        self.conv_feat = BranchEEGNetTransformer(
            in_channels=self.in_channels,
            f1=f1,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            pooling_size1=pooling_size1,
            pooling_size2=pooling_size2,
            dropout=dropout,
        )
        self.position = PositionalEncoding(d_model)
        self.trans_feat = TransformerEncoder(n_heads, depth, d_model)

        self.flatten = nn.Flatten()
        fc_input_dim = self._calculate_fc_input_dim()
        self.classification = ClassificationHead(
            fc_input_dim,
            self.number_class
        )

    def _calculate_fc_input_dim(self):
        dummy_input = torch.zeros(1, 1, self.in_channels, self.n_samples)
        with torch.no_grad():
            dummy_output = self.conv_feat(dummy_input)
        return dummy_output.reshape(1, -1).size(1)

    def forward(self, inp):
        conv_feat = self.conv_feat(inp)
        conv_feat = conv_feat * math.sqrt(self.d_model)
        conv_feat = self.position(conv_feat)
        trans_feat = self.trans_feat(conv_feat)
        features = conv_feat + trans_feat
        out = self.classification(self.flatten(features))
        return out


if __name__ == '__main__':
    from thop import profile

    model = CTNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
