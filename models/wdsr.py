from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn

from .srmodel import SRModel


class _Block_A(nn.Module):
    def __init__(
            self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(_Block_A, self).__init__()
        self.res_scale = res_scale
        block_feats = 4 * n_feats
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class _Block_B(nn.Module):
    def __init__(
            self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(_Block_B, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class WDSR(SRModel):
    """
    LightningModule for WDSR, https://bmvc2019.org/wp-content/uploads/papers/0288-paper.pdf.
    """
    @staticmethod
    def add_model_specific_args(parent: ArgumentParser) -> ArgumentParser:
        parent = SRModel.add_model_specific_args(parent)
        parser = ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--n_feats', type=int, default=128,
                            help='number of features to use in conv')
        parser.add_argument('--n_resblocks', type=int, default=16,
                            help='number of residual blocks')
        parser.add_argument('--res_scale', type=float, default=1,
                            help='residual scaling')
        parser.add_argument('--type', type=str, default='B',
                            choices=['A', 'B'])
        return parser

    def __init__(self, args: Namespace):
        super(WDSR, self).__init__(args)
        kernel_size = 3

        def wn(x): return nn.utils.weight_norm(x)

        # computed on training images of DIV2K dataset
        self.rgb_mean = torch.FloatTensor(
            [0.4488, 0.4371, 0.4040]).view([1, 3, 1, 1])

        head = []
        head.append(
            wn(nn.Conv2d(3, args.n_feats, 3, padding=3//2)))

        body = []

        if args.type == 'A':
            block = _Block_A
        else:  # if args.type == 'B':
            block = _Block_B

        for i in range(args.n_resblocks):
            body.append(
                block(args.n_feats, kernel_size, act=nn.ReLU(True), res_scale=args.res_scale, wn=wn))

        tail = []
        out_feats = self._scale_factor * self._scale_factor * 3
        tail.append(
            wn(nn.Conv2d(args.n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(self._scale_factor))

        skip = []
        skip.append(
            wn(nn.Conv2d(3, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(self._scale_factor))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        self.rgb_mean = self.rgb_mean.to(self.device)
        x = x - self.rgb_mean

        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)

        x = self.tail(x)
        x += s

        x = x + self.rgb_mean

        return x
