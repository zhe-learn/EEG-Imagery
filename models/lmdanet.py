"""
Miao, Z., Zhao, M., Zhang, X., & Ming, D. (2023).
LMDA-Net: A lightweight multi-dimensional attention network for general EEG-based brain-computer interfaces and interpretability.
NeuroImage, 276, 120209.

"""

import torch
import torch.nn as nn


class LMDANet(nn.Module):
    def __init__(
            self,
            in_channels=22,
            n_classes=4,
            n_samples=1000,
            depth=9,
            kernel_size=75,
            channel_depth1=24,
            channel_depth2=9,
            avg_pool_size=25
    ):
        super().__init__()
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, in_channels), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel_size), groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(in_channels, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avg_pool_size)),
            nn.Dropout(p=0.65),
        )

        out = torch.ones((1, 1, in_channels, n_samples))
        out = torch.einsum("bdcw, hdc->bhcw", out, self.channel_weight)
        out = self.time_conv(out)
        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape

        self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], n_classes)

    def EEGDepthAttention(self, inp):
        batch_size, channels, height, width = inp.size()
        kernel_size = 7
        adaptive_pool = nn.AdaptiveAvgPool2d((1, width))
        conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=True
        ).to(inp.device)
        nn.init.xavier_uniform_(conv.weight)
        nn.init.constant_(conv.bias, 0)
        softmax = nn.Softmax(dim=-2)
        pooled = adaptive_pool(inp)
        transposed = pooled.transpose(-2, -3)
        gated = conv(transposed)
        gated = softmax(gated)
        gated = gated.transpose(-2, -3)
        return gated * channels * inp

    def forward(self, inp):
        inp = torch.einsum("bdcw, hdc->bhcw", inp, self.channel_weight)
        inp = self.time_conv(inp)
        inp = self.EEGDepthAttention(inp)
        inp = self.chanel_conv(inp)
        inp = self.norm(inp)
        inp = torch.flatten(inp, 1)
        inp = self.classifier(inp)
        return inp


if __name__ == '__main__':
    from thop import profile

    model = LMDANet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
