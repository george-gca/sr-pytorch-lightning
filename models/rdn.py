from typing import Any

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
    def __init__(self, rdn_config: str='B', G0: int=64, kernel_size: int=3, **kwargs: dict[str, Any]):
        super(RDN, self).__init__(**kwargs)

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[rdn_config]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(
            self._channels, G0, kernel_size, padding=(kernel_size-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kernel_size, padding=(
            kernel_size-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self._RDBs = nn.ModuleList()
        for i in range(self.D):
            self._RDBs.append(
                _RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kernel_size, padding=(
                kernel_size-1)//2, stride=1)
        ])

        # Up-sampling net
        if self._scale_factor == 2 or self._scale_factor == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * self._scale_factor * self._scale_factor, kernel_size,
                          padding=(kernel_size-1)//2, stride=1),
                nn.PixelShuffle(self._scale_factor),
                nn.Conv2d(G, 3, kernel_size,
                          padding=(kernel_size-1)//2, stride=1)
            ])
        elif self._scale_factor == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kernel_size,
                          padding=(kernel_size-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kernel_size,
                          padding=(kernel_size-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, self._channels, kernel_size,
                          padding=(kernel_size-1)//2, stride=1)
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
