from typing import Any

import torch.nn as nn

from .common import DefaultConv2d, MeanShift, ResBlock, UpscaleBlock
from .srmodel import SRModel


class EDSR(SRModel):
    """
    LightningModule for EDSR, https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf.
    """
    def __init__(self, n_feats: int=64, n_resblocks: int=16, res_scale: int=1, **kwargs: dict[str, Any]):
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
