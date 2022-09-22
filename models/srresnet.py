from argparse import ArgumentParser, Namespace

import torch.nn as nn

from .common import BasicBlock, DefaultConv2d, ResBlock, UpscaleBlock
from .srmodel import SRModel


class SRResNet(SRModel):
    """
    LightningModule for SRResNet, https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf.
    """
    @staticmethod
    def add_model_specific_args(parent: ArgumentParser) -> ArgumentParser:
        parent = SRModel.add_model_specific_args(parent)
        parser = ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--n_resblocks', type=int, default=16,
                            help='number of residual blocks')
        parser.add_argument('--n_feats', type=int, default=64,
                            help='number of feature maps')
        return parser

    def __init__(self, args: Namespace):
        super(SRResNet, self).__init__(args)

        self.head = BasicBlock(
            in_channels=3, out_channels=args.n_feats, kernel_size=9, act=nn.PReLU())

        m_body = [
            ResBlock(n_feats=args.n_feats, kernel_size=3,
                     n_conv_layers=2, norm=nn.BatchNorm2d(args.n_feats), act=nn.PReLU()) for _ in range(args.n_resblocks)
        ]
        m_body.append(BasicBlock(
            in_channels=args.n_feats, out_channels=args.n_feats, kernel_size=3, norm=nn.BatchNorm2d(args.n_feats), act=None))
        self.body = nn.Sequential(*m_body)

        m_tail = [
            UpscaleBlock(
                self._scale_factor, n_feats=args.n_feats, act=nn.PReLU()),
            DefaultConv2d(in_channels=args.n_feats,
                          out_channels=3, kernel_size=9)
        ]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x
