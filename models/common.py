from math import log2

import torch
from torch import nn


class DefaultConv2d(nn.Conv2d):
    """
    Conv2d that keeps the height and width of the input in the output by default
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        padding: str | int | tuple[int, int] = 'same',
        **kwargs
    ):
        if isinstance(padding, str):
            lower_padding = padding.lower()
            assert(lower_padding == 'valid' or lower_padding == 'same')
            if lower_padding == 'valid':
                padding = 0
            else:  # if lower_padding == 'same':
                if isinstance(kernel_size, int):
                    padding = kernel_size // 2
                else:
                    padding = tuple(k // 2 for k in kernel_size)

        super(DefaultConv2d, self).__init__(
            kernel_size=kernel_size, padding=padding, **kwargs)


class BasicBlock(nn.Sequential):
    """
    Block composed of a Conv2d with normalization and activation function when given
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 3,
        bias: bool = True,
        conv: nn.Module = DefaultConv2d,
        norm: nn.Module = None,
        act: nn.Module = nn.ReLU(True)
    ):
        m = [conv(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, bias=bias)]
        if norm is not None:
            m.append(norm)
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range: int = 1,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        rgb_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sign: int = -1
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    """
    ResBlock composed of a sequence of Conv2d followed by normalization when given
    and activation function when given, except for the last conv. It multiplies
    its output with res_scale value, and has a residual connection with its input
    """

    def __init__(
        self,
        conv: nn.Module = DefaultConv2d,
        n_feats: int = 64,
        kernel_size: int = 3,
        n_conv_layers: int = 2,
        bias: bool = True,
        norm: nn.Module = None,
        act: nn.Module = nn.ReLU(True),
        res_scale: float = 1.
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(n_conv_layers):
            m.append(conv(in_channels=n_feats, out_channels=n_feats,
                          kernel_size=kernel_size, bias=bias))
            if norm is not None:
                m.append(norm)
            if act is not None and i < n_conv_layers - 1:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x

        return res


class UpscaleBlock(nn.Sequential):
    """
    Upscale block using sub-pixel convolutions.
    `scale_factor` can be selected from {2, 3, 4, 8}.
    """

    def __init__(
        self,
        scale_factor: int = 4,
        n_feats: int = 64,
        kernel_size=3,
        act: nn.Module = None
    ):
        assert scale_factor in {2, 3, 4, 8}

        layers = []
        for _ in range(int(log2(scale_factor))):
            r = 2 if scale_factor % 2 == 0 else 3
            layers += [
                DefaultConv2d(in_channels=n_feats, out_channels=n_feats * r * r,
                              kernel_size=kernel_size),
                nn.PixelShuffle(r),
            ]

            if act is not None:
                layers.append(act)

        super(UpscaleBlock, self).__init__(*layers)
