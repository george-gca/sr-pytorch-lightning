from argparse import ArgumentParser
from typing import Any, Dict

import torch.nn as nn

from .common import DefaultConv2d, MeanShift, ResBlock, UpscaleBlock
from .srmodel import SRModel


class EDSR(SRModel):
    """
    LightningModule for EDSR, https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf.
    """
    @staticmethod
    def add_model_specific_args(parent: ArgumentParser) -> ArgumentParser:
        parent = SRModel.add_model_specific_args(parent)
        parser = ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--n_feats', type=int, default=64,
                            help='number of feature maps')
        parser.add_argument('--n_resblocks', type=int, default=16,
                            help='number of residual blocks')
        parser.add_argument('--res_scale', type=float, default=1,
                            help='residual scaling')
        return parser

    def __init__(self, n_feats: int=64, n_resblocks: int=16, res_scale: int=1, **kwargs: Dict[str, Any]):
        super(EDSR, self).__init__(**kwargs)
        kernel_size = 3

        if self._channels == 3:
            self.sub_mean = MeanShift()
            self.add_mean = MeanShift(sign=1)

        m_head = [DefaultConv2d(in_channels=self._channels,
                                out_channels=n_feats, kernel_size=kernel_size)]

        m_body = [
            ResBlock(n_feats=n_feats, kernel_size=kernel_size, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        m_body.append(DefaultConv2d(in_channels=n_feats,
                                    out_channels=n_feats, kernel_size=kernel_size))

        m_tail = [
            UpscaleBlock(self._scale_factor, n_feats),
            DefaultConv2d(in_channels=n_feats, out_channels=self._channels,
                          kernel_size=kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if self._channels == 3:
            x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        if self._channels == 3:
            x = self.add_mean(x)

        return x
