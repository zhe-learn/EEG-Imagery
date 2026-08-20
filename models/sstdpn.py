"""
Han, channels., Liu, channels., Wang, J., Wang, Y., Cai, channels., & Qian, D. (2025).
A spatial–spectral and temporal dual prototype network for motor imagery brain–computer interface.
Knowledge-Based Systems, 315, 113315.

"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import rearrange


class PrototypeLoss(nn.Module):
    def forward(self, features, proxy, labels):
        label_prototypes = torch.index_select(proxy, dim=0, index=labels)

        pl = huber_loss(features, label_prototypes, sigma=1)
        pl_loss = torch.mean(pl)

        return pl_loss


def huber_loss(inp, target, sigma=1):
    beta = 1.0 / (sigma ** 2)
    diff = torch.abs(inp - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    return torch.sum(loss, dim=1)


class NormIncreaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mat):
        norms = torch.norm(mat, p=2, dim=1)
        loss = -norms
        return loss.mean()


class LightweightConv1d(nn.Module):
    def __init__(
            self,
            in_channels,
            num_heads=1,
            depth_multiplier=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            weight_softmax=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads * depth_multiplier, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * depth_multiplier))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp):
        inp = inp.squeeze(1)
        batch_size, channels, seq_len = inp.size()
        n_heads = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = nnf.softmax(weight, dim=-1)

        inp = rearrange(inp, "b (h c) t ->(b c) h t", h=n_heads)
        if self.bias is None:
            output = nnf.conv1d(
                inp,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        else:
            output = nnf.conv1d(
                inp,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        output = rearrange(output, "(b c) h t ->b (h c) t", b=batch_size)

        return output


class VarMaxPool1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, inp):
        mean_of_squares = nnf.avg_pool1d(inp ** 2, self.kernel_size, self.stride, self.padding)
        square_of_mean = (nnf.avg_pool1d(inp, self.kernel_size, self.stride, self.padding) ** 2)
        variance = mean_of_squares - square_of_mean
        out = nnf.avg_pool1d(variance, variance.shape[-1])

        return out


class VarPool1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, inp):
        mean_of_squares = nnf.avg_pool1d(inp ** 2, self.kernel_size, self.stride, self.padding)
        square_of_mean = (nnf.avg_pool1d(inp, self.kernel_size, self.stride, self.padding) ** 2)
        variance = mean_of_squares - square_of_mean

        return variance


class SSA(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode="var", after_relu=False):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        self.GP = VarMaxPool1D(250)

    def forward(self, inp):

        if self.mode == "l2":
            embedding = (inp.pow(2).sum(2, keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == "l1":
            if not self.after_relu:
                _x = torch.abs(inp)
            else:
                _x = inp
            embedding = _x.sum((2), keepdim=True)
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        elif self.mode == "var":

            embedding = (self.GP(inp) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta)

        return inp * gate, gate


class Mixer1D(nn.Module):
    def __init__(self, dim, kernel_sizes=[50, 100, 250]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self.seq_len = len(kernel_sizes)
        for kernel in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=kernel, stride=int(kernel / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, inp):
        batch_size, feat_dim, seq_len = inp.shape
        split = torch.split(inp, feat_dim // self.seq_len, dim=1)
        out = []
        for idx in range(len(split)):
            inp = self.var_layers[idx](split[idx])
            out.append(inp)
        out = torch.concat(out, dim=1)
        return out


class Efficient_Encoder(nn.Module):
    def __init__(
            self,
            samples,
            chans,
            f1,
            f2,
            time_kernel_length,
            pool_kernels,
    ):
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=f1,
            kernel_size=time_kernel_length,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(chans * f1)

        self.chanConv = nn.Sequential(
            nn.Conv1d(
                chans * f1,
                f2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(f2),
            nn.ELU(),
        )

        self.mixer = Mixer1D(dim=f2, kernel_sizes=pool_kernels)

    def forward(self, inp):
        inp = self.time_conv(inp)
        inp, _ = self.ssa(inp)
        chan_feat = self.chanConv(inp)

        feature = self.mixer(chan_feat)

        return feature


class SSTDPN(nn.Module):
    def __init__(
            self,
            in_channels=22,
            n_classes=4,
            n_samples=1000,
            f1=9,
            f2=48,
            time_kernel_length=75,
            pool_kernels=[50, 100, 200],
    ):
        """
        time_kernel_length=50,
        pool_kernels=[50, 100, 150],
        """
        super().__init__()
        self.encoder = Efficient_Encoder(
            samples=n_samples,
            chans=in_channels,
            f1=f1,
            f2=f2,
            time_kernel_length=time_kernel_length,
            pool_kernels=pool_kernels,
        )
        self.features = None

        inp = torch.ones((1, in_channels, n_samples))
        out = self.encoder(inp)
        feat_dim = out.shape[-1]

        self.isp = nn.Parameter(torch.randn(n_classes, feat_dim), requires_grad=True)
        self.icp = nn.Parameter(torch.randn(n_classes, feat_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.isp)

    def get_features(self):
        if self.features is not None:
            return self.features
        else:
            raise RuntimeError("No features available. Run forward() first.")

    def forward(self, inp):

        features = self.encoder(inp)
        self.features = features
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)
        logits = torch.einsum("bd,cd->bc", features, self.isp)

        return logits


if __name__ == '__main__':
    from thop import profile

    model = SSTDPN().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
