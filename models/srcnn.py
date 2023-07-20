from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from .srmodel import SRModel


class SRCNN(SRModel):
    """
    LightningModule for SRCNN, https://ieeexplore.ieee.org/document/7115171?arnumber=7115171
    https://arxiv.org/pdf/1501.00092.pdf.
    """
    def __init__(self, **kwargs: dict[str, Any]):
        super(SRCNN, self).__init__(**kwargs)
        self._net = nn.Sequential(
            nn.Conv2d(self._channels, 64, 9, padding=4),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, self._channels, 5, padding=2)
        )

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self._scale_factor, mode='bicubic')
        return self._net(x)
