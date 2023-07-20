from typing import Any

import torch.nn as nn

from .common import BasicBlock, DefaultConv2d, ResBlock, UpscaleBlock
from .srmodel import SRModel


class SRResNet(SRModel):
    def __init__(self, n_resblocks: int=16, n_feats: int=64, **kwargs: dict[str, Any]):
        super(SRResNet, self).__init__(**kwargs)

        self.head = BasicBlock(
            in_channels=self._channels, out_channels=n_feats, kernel_size=9, act=nn.PReLU())

        m_body = [
            ResBlock(n_feats=n_feats, kernel_size=3,
                     n_conv_layers=2, norm=nn.BatchNorm2d(n_feats), act=nn.PReLU()) for _ in range(n_resblocks)
        ]
        m_body.append(BasicBlock(
            in_channels=n_feats, out_channels=n_feats, kernel_size=3, norm=nn.BatchNorm2d(n_feats), act=None))
        self.body = nn.Sequential(*m_body)

        m_tail = [
            UpscaleBlock(
                self._scale_factor, n_feats=n_feats, act=nn.PReLU()),
            DefaultConv2d(in_channels=n_feats,
                          out_channels=self._channels, kernel_size=9)
        ]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x
