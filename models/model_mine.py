"""
Prototype Discriminative Channel-Periodicity Aware Framework for MI-EEG Generalization
Inspired by: TimesNet, FACT, SST-DPN

"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf


class PrototypeDiscriminativeLoss(nn.Module):
    def __init__(self, alpha, beta, margin=1.0):
        super().__init__()
        self.smooth_loss = nn.SmoothL1Loss(reduction='none')
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, features, proxy, labels):
        batch_size, feat_dim = features.shape
        num_classes = proxy.shape[0]

        label_prototypes = torch.index_select(proxy, dim=0, index=labels)
        att_loss = self.smooth_loss(features, label_prototypes)
        att_loss = torch.sum(att_loss, dim=1).mean()

        mask = (torch.arange(num_classes, device=features.device) != labels.unsqueeze(1)).float()

        features_expand = features.unsqueeze(1).expand(-1, num_classes, -1)
        proxy_expand = proxy.unsqueeze(0).expand(batch_size, -1, -1)
        all_distances = torch.sum(self.smooth_loss(features_expand, proxy_expand), dim=2)  # [batch_size, num_classes]

        rep_loss = torch.clamp(self.margin - all_distances, min=0.0)
        rep_loss = torch.sum(rep_loss * mask, dim=1).mean()

        return att_loss * self.alpha + rep_loss * self.beta


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


class ResidualDualChannelAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        kernel = kernel_size if kernel_size % 2 else kernel_size + 1
        pad = kernel // 2

        self.K1 = nn.Conv1d(1, 1, kernel, padding=pad, bias=False)
        self.K2 = nn.Conv1d(1, 1, kernel, padding=pad, bias=False)
        self.K3 = nn.Conv1d(1, 1, kernel, padding=pad, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, inp):
        gap = nnf.adaptive_avg_pool1d(inp, 1)
        gap_transposed = gap.transpose(1, 2)

        gate_1 = self.sigmoid(self.K1(gap_transposed))
        gate_2 = self.sigmoid(self.K2(gap_transposed))
        gate_3 = self.sigmoid(self.K3(gap_transposed))

        psi_1_T = gate_1.transpose(1, 2)
        local_out = inp * psi_1_T

        psi_3_T = gate_3.transpose(1, 2)
        attention_map = self.softmax(torch.matmul(psi_3_T, gate_2))
        global_out = torch.matmul(attention_map, inp)

        out = local_out + global_out + inp
        # out = 0.0001 * local_out + 0.0001 * global_out + inp

        return out


class FastFourierTransformationForPeriod(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, inp):
        fft_signal = torch.fft.rfft(inp, dim=1)
        frequency_list = abs(fft_signal).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, int(self.top_k))
        top_list = top_list.detach().cpu().numpy()
        length = top_list.shape[0]
        for idx in range(length - 1, 0, -1):
            top_list[idx] = top_list[idx - 1]
        top_list[0] = 1

        period = inp.shape[1] // top_list
        return period, abs(fft_signal).mean(-1)[:, top_list]


class Conv2DInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_kernels):
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


class PeriodicityAwareModule(nn.Module):
    def __init__(self, seq_len, d_model, n_kernels, top_k):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.fft_get_p = FastFourierTransformationForPeriod(top_k)
        self.d_model = d_model
        self.conv = nn.Sequential(
            Conv2DInceptionBlock(self.d_model, self.d_model, n_kernels),
            nn.GELU(),
            Conv2DInceptionBlock(self.d_model, self.d_model, n_kernels)
        )

    def forward(self, inp):
        batch_size, seq_len, n_features = inp.size()
        period_list, period_weight = self.fft_get_p(inp)
        # print("period_list:", period_list)
        # print("period_weight:", period_weight)
        res = []
        for idx in range(self.top_k):
            period = period_list[idx]
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                # print("length:", length)
                padding = torch.zeros([inp.shape[0], (length - self.seq_len), inp.shape[2]]).to(inp.device)
                # print("padding:", padding)
                out = torch.cat([inp, padding], dim=1)
            else:
                length = self.seq_len
                out = inp
            out = out.reshape(batch_size, length // period, period, n_features).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, n_features)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = nnf.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, seq_len, n_features, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + inp
        return res


class EEGBackbone(nn.Module):
    def __init__(
            self,
            num_channels,
            f1,
            depth_multiplier,
            f2,
            dropout,
            temp_conv_kernel_size,
            pool_size1,
            pool_size2,
            pool_stride1,
            pool_stride2
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, temp_conv_kernel_size), padding="same", bias=False),  # 16
            nn.BatchNorm2d(f1),

            Conv2dWithConstraint(
                in_channels=f1,
                out_channels=f1 * depth_multiplier,
                kernel_size=(num_channels, 1),
                padding='valid',
                groups=f1,
                bias=False,
                max_norm=1.0
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_size1), stride=(1, pool_stride1)),  # 64, 8
            nn.Dropout(p=dropout),

            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f1 * depth_multiplier,
                kernel_size=(1, 16),
                groups=f1 * depth_multiplier,
                padding='same',
                bias=False
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_size2), stride=(1, pool_stride2)),  # 2, 2
            nn.Dropout(p=dropout)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, inp):
        return self.backbone(inp)


class Net(nn.Module):
    def __init__(
            self,
            in_channels=22,
            n_classes=4,
            n_samples=1000,
            f1=16,
            depth_multiplier=2,
            f2=32,
            temp_conv_kernel_size=64,
            pool_size1=8,
            pool_size2=8,
            pool_stride1=8,
            pool_stride2=8,
            dropout=0.3,
            n_kernels=4,
            n_repeats=1,
            top_k=2
    ):
        super().__init__()

        self.n_repeats = n_repeats
        self.num_channels = in_channels
        self.num_classes = n_classes
        self.num_time_points = n_samples

        self.backbone = EEGBackbone(
            num_channels=in_channels,
            f1=f1,
            depth_multiplier=depth_multiplier,
            f2=f2,
            dropout=dropout,
            temp_conv_kernel_size=temp_conv_kernel_size,
            pool_size1=pool_size1,
            pool_size2=pool_size2,
            pool_stride1=pool_stride1,
            pool_stride2=pool_stride2
        )

        self.attention = ResidualDualChannelAttention()
        self.layer_norm = nn.LayerNorm(f2)
        self.periodicity_aware = nn.ModuleList([
            PeriodicityAwareModule(
                seq_len=self._forward_backbone().shape[-1],
                d_model=f2,
                n_kernels=n_kernels,
                top_k=top_k
            ) for _ in range(n_repeats)
        ])

        linear_in = self._forward_flatten().shape[1]
        self.features = None
        self.proxy = nn.Parameter(torch.randn(n_classes, linear_in), requires_grad=True)

        self.head = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(linear_in, n_classes, max_norm=1.0)
        )

    def _forward_backbone(self):
        inp = torch.randn(1, 1, self.num_channels, self.num_time_points)
        inp = self.backbone(inp)
        return inp

    def _forward_flatten(self):
        inp = self._forward_backbone()
        # inp = self.attention(inp.squeeze(dim=2))
        # inp = inp.squeeze(dim=2).permute(0, 2, 1)
        # for idx in range(self.n_repeats):
        #     inp = self.layer_norm(self.periodicity_aware[idx](inp))
        inp = inp.flatten(start_dim=1)
        return inp

    def forward(self, inp):
        inp = self.backbone(inp)
        inp = self.attention(inp.squeeze(dim=2))
        inp = inp.squeeze(dim=2).permute(0, 2, 1)
        for idx in range(self.n_repeats):
            inp = self.layer_norm(self.periodicity_aware[idx](inp))

        feat = inp.flatten(start_dim=1)
        self.features = feat

        inp = self.head(feat)
        return inp


if __name__ == '__main__':
    from thop import profile

    model = Net().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    inp = torch.randn(1, 1, 22, 1000, ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    flops, _ = profile(model, inputs=(inp,))
    print('Trainable Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('FLOPs: ' + str(flops / 1000000.0))
