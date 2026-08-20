"""
LDJ, S. R. S. J. F. Glasstetter M. Eggensperger K. Tangermann M. et al.(2017).
Deep learning with convolutional neural networks for EEG decoding and visualization.
Hum. Brain Mapp, 38, 5391-5420.

"""

import torch

from torch import nn


class DeepConvNet(nn.Module):
    def __init__(
            self,
            in_channels=22,
            n_classes=4,
            n_samples=1000,
            dropout=0.5,
            pool_size=3,
            pool_stride=3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_samples = n_samples

        self.deepnet = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (in_channels, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3), (1, 3)),
            nn.Dropout(dropout),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3), (1, 3)),
            nn.Dropout(dropout),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, pool_size), (1, pool_stride)),  # 2, 2
            nn.Dropout(dropout),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 3), (1, 3)),
            nn.Dropout(dropout),
        )

        self.flatten = nn.Flatten()
        fc_input_dim = self._calculate_fc_input_dim()
        self.classifier = nn.Linear(fc_input_dim, n_classes)

    def _calculate_fc_input_dim(self):
        dummy_input = torch.zeros(1, 1, self.in_channels, self.n_samples)
        with torch.no_grad():
            dummy_output = self.deepnet(dummy_input)
        return dummy_output.reshape(1, -1).size(1)

    def forward(self, inp):
        inp = self.deepnet(inp)
        inp = self.flatten(inp)
        inp = self.classifier(inp)
        return inp


if __name__ == '__main__':
    from thop import profile

    model = DeepConvNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
