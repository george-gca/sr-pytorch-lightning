from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn

from .srmodel import SRModel


class _RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(_RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class _RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(_RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(_RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(SRModel):
    """
    LightningModule for RDN, https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Residual_Dense_Network_CVPR_2018_paper.pdf.
    """
    @staticmethod
    def add_model_specific_args(parent: ArgumentParser) -> ArgumentParser:
        parent = SRModel.add_model_specific_args(parent)
        parser = ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--G0', type=int, default=64)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--rdn_config', type=str, default='B',
                            choices=['A', 'B'])
        return parser

    def __init__(self, args: Namespace):
        super(RDN, self).__init__(args)

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.rdn_config]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(
            3, args.G0, args.kernel_size, padding=(args.kernel_size-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(args.G0, args.G0, args.kernel_size, padding=(
            args.kernel_size-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self._RDBs = nn.ModuleList()
        for i in range(self.D):
            self._RDBs.append(
                _RDB(growRate0=args.G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * args.G0, args.G0, 1, padding=0, stride=1),
            nn.Conv2d(args.G0, args.G0, args.kernel_size, padding=(
                args.kernel_size-1)//2, stride=1)
        ])

        # Up-sampling net
        if self._scale_factor == 2 or self._scale_factor == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(args.G0, G * self._scale_factor * self._scale_factor, args.kernel_size,
                          padding=(args.kernel_size-1)//2, stride=1),
                nn.PixelShuffle(self._scale_factor),
                nn.Conv2d(G, 3, args.kernel_size,
                          padding=(args.kernel_size-1)//2, stride=1)
            ])
        elif self._scale_factor == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(args.G0, G * 4, args.kernel_size,
                          padding=(args.kernel_size-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, args.kernel_size,
                          padding=(args.kernel_size-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, 3, args.kernel_size,
                          padding=(args.kernel_size-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self._RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        return self.UPNet(x)
