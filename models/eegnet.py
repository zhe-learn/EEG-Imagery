"""
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.
Journal of neural engineering, 15(5), 056013.

"""

import torch
from torch import nn


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


class EEGNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 22,
            n_classes: int = 4,
            n_samples: int = 1000,
            f1: int = 8,
            depth_multiplier: int = 2,
            f2: int = 16,
            kernel_length: int = 64,
            dropout: float = 0.5,
            kernel_length_dw_sep: int = 16
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_samples = n_samples

        self.eeg = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), bias=False, padding="same"),
            nn.BatchNorm2d(f1),

            Conv2dWithConstraint(f1, f1 * depth_multiplier, (in_channels, 1), bias=False, groups=f1, max_norm=1),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),

            nn.Conv2d(f2, f2, (1, kernel_length_dw_sep), bias=False, groups=f2, padding="same"),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        fc_input_dim = self._calculate_fc_input_dim()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, n_classes)
        )

    def _calculate_fc_input_dim(self):
        dummy_input = torch.zeros(1, 1, self.in_channels, self.n_samples)
        with torch.no_grad():
            dummy_output = self.eeg(dummy_input)
        return dummy_output.reshape(1, -1).size(1)

    def forward(self, inp):
        inp = self.eeg(inp)
        inp = self.classifier(inp)
        return inp


if __name__ == '__main__':
    from thop import profile

    model = EEGNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
