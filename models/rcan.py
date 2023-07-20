from typing import Any

import torch.nn as nn

from .common import DefaultConv2d, MeanShift, UpscaleBlock
from .srmodel import SRModel


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = torch.Size([-1, 64, 32, 32])
        y = self.avg_pool(x)
        # # y = torch.Size([-1, 64, 1, 1])
        y = self.conv_du(y)
        # # y = torch.Size([-1, 64, 1, 1])
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(
                in_channels=n_feat, out_channels=n_feat, kernel_size=kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
            for _ in range(n_resblocks)]
        modules_body.append(
            conv(in_channels=n_feat, out_channels=n_feat, kernel_size=kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RCAN(SRModel):
    """
    LightningModule for RCAN, https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf.
    """
    def __init__(self, n_feats: int=64, n_resblocks: int=16, n_resgroups: int=10, reduction: int=16, res_scale: int=1, **kwargs: dict[str, Any]):
        super(RCAN, self).__init__(**kwargs)
        kernel_size = 3

        if self._channels == 3:
            # RGB mean for DIV2K
            self.sub_mean = MeanShift()

        # define head module
        modules_head = [DefaultConv2d(in_channels=self._channels,
                             out_channels=n_feats, kernel_size=kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                DefaultConv2d, n_feats, kernel_size, reduction, act=nn.ReLU(True), res_scale=res_scale, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]

        modules_body.append(
            DefaultConv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size))

        # define tail module
        modules_tail = [
            UpscaleBlock(self._scale_factor, n_feats),
            DefaultConv2d(in_channels=n_feats, out_channels=self._channels, kernel_size=kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        if self._channels == 3:
            self.add_mean = MeanShift(sign=1)

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
