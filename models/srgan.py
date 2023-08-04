from math import ceil, sqrt
from typing import Any

import piq
import torch
import torch.nn as nn
import torch.optim as optim
from kornia.color import rgb_to_grayscale
from losses.losses import GANLoss, TVLoss, VGGLoss
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid

from models.srmodel import SRModel

from .common import UpscaleBlock


class _SRResNet(nn.Module):
    """
    PyTorch Module for SRGAN, https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf.
    """

    def __init__(self, scale_factor=4, ngf=64, n_blocks=16):
        super(_SRResNet, self).__init__()

        self._head = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(self._channels, ngf, kernel_size=9),
            nn.PReLU()
        )
        self._body = nn.Sequential(
            *[_SRGANBlock(ngf) for _ in range(n_blocks)],
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3),
            nn.BatchNorm2d(ngf)
        )
        self._tail = nn.Sequential(
            UpscaleBlock(scale_factor, ngf, act=nn.PReLU),
            nn.ReflectionPad2d(4),
            nn.Conv2d(ngf, self._channels, kernel_size=9),
            nn.Tanh()
        )

    def forward(self, x):
        x = self._head(x)
        x = self._body(x) + x
        x = self._tail(x)
        return (x + 1) / 2


class _SRGANBlock(nn.Module):
    """
    Building block of SRGAN.
    """

    def __init__(self, dim):
        super(_SRGANBlock, self).__init__()
        self._net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self._net(x)


class _Discriminator(nn.Sequential):
    """
    _Discriminator for SRGAN.
    Dense layers are replaced with global poolings and 1x1 convolutions.
    """

    def __init__(self, ndf):

        def ConvBlock(in_channels, out_channels, stride):
            out = [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(out_channels),
            ]
            return out

        super(_Discriminator, self).__init__(
            nn.Conv2d(3, ndf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

            *ConvBlock(ndf, ndf, 2),

            *ConvBlock(ndf, ndf * 2, 1),
            *ConvBlock(ndf * 2, ndf * 2, 2),

            *ConvBlock(ndf * 2, ndf * 4, 1),
            *ConvBlock(ndf * 4, ndf * 4, 2),

            *ConvBlock(ndf * 4, ndf * 8, 1),
            *ConvBlock(ndf * 8, ndf * 8, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )


class SRGAN(SRModel):
    """
    LightningModule for SRGAN, https://arxiv.org/pdf/1609.04802.
    """
    def __init__(self, ngf: int=64, ndf: int=64, n_blocks: int=16, **kwargs: dict[str, Any]):

        super(SRGAN, self).__init__(**kwargs)

        # networks
        self._net_G = _SRResNet(self._scale_factor, ngf, n_blocks)
        self._net_D = _Discriminator(ndf)

        # training criterions
        self._criterion_MSE = nn.MSELoss()
        self._criterion_VGG = VGGLoss(net_type='vgg19', layer='relu5_4')
        self._criterion_GAN = GANLoss(gan_mode='wgangp')
        self._criterion_TV = TVLoss()

        # validation metrics
        self._criterion_PSNR = piq.psnr()
        self._criterion_SSIM = piq.ssim()

    def forward(self, input):
        return self._net_G(input)

    def training_step(self, batch, batch_idx, optimizer_idx):
        img_lr = batch['lr']  # \in [0, 1]
        img_hr = batch['hr']  # \in [0, 1]

        if optimizer_idx == 0:  # train _Discriminator
            self.img_sr = self.forward(img_lr)  # \in [0, 1]

            # for real image
            d_out_real = self._net_D(img_hr)
            d_loss_real = self._criterion_GAN(d_out_real, True)
            # for fake image
            d_out_fake = self._net_D(self.img_sr.detach())
            d_loss_fake = self._criterion_GAN(d_out_fake, False)

            # combined _Discriminator loss
            d_loss = 1 + d_loss_real + d_loss_fake

            return {'loss': d_loss, 'prog': {'tng/d_loss': d_loss}}

        elif optimizer_idx == 1:  # train generator
            # content loss
            mse_loss = self._criterion_MSE(self.img_sr * 2 - 1,  # \in [-1, 1]
                                           img_hr * 2 - 1)  # \in [-1, 1]
            vgg_loss = self._criterion_VGG(self.img_sr, img_hr)
            content_loss = (vgg_loss + mse_loss) / 2
            # adversarial loss
            adv_loss = self._criterion_GAN(self._net_D(self.img_sr), True)
            # tv loss
            tv_loss = self._criterion_TV(self.img_sr)

            # combined generator loss
            g_loss = content_loss + 1e-3 * adv_loss + 2e-8 * tv_loss

            if self.global_step % self.trainer.row_log_interval == 0:
                nrow = ceil(sqrt(self._batch_size))
                self.logger.experiment.add_image(
                    tag='train/lr_img',
                    img_tensor=make_grid(img_lr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    tag='train/hr_img',
                    img_tensor=make_grid(img_hr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )
                self.logger.experiment.add_image(
                    tag='train/sr_img',
                    img_tensor=make_grid(self.img_sr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )

            return {'loss': g_loss, 'prog': {'tng/g_loss': g_loss,
                                             'tng/content_loss': content_loss,
                                             'tng/adv_loss': adv_loss,
                                             'tng/tv_loss': tv_loss}}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        with torch.no_grad():
            img_lr = batch['lr']
            img_hr = batch['hr']
            img_sr = self.forward(img_lr)

            img_hr_ = rgb_to_grayscale(img_hr)
            img_sr_ = rgb_to_grayscale(img_sr)

            psnr = self._criterion_PSNR(img_sr_, img_hr_)
            ssim = 1 - self._criterion_SSIM(img_sr_, img_hr_)  # invert

        return {'psnr': psnr, 'ssim': ssim}

    def validation_end(self, outputs):
        val_psnr_mean = 0
        val_ssim_mean = 0
        for output in outputs:
            val_psnr_mean += output['psnr']
            val_ssim_mean += output['ssim']
        val_psnr_mean /= len(outputs)
        val_ssim_mean /= len(outputs)
        return {'val/psnr': val_psnr_mean.item(),
                'val/ssim': val_ssim_mean.item()}

    def configure_optimizers(self):
        optimizer_G = optim.Adam(self._net_G.parameters(), lr=1e-4)
        optimizer_D = optim.Adam(self._net_D.parameters(), lr=1e-4)
        scheduler_G = StepLR(optimizer_G, step_size=1e+5, gamma=0.1)
        scheduler_D = StepLR(optimizer_D, step_size=1e+5, gamma=0.1)
        return [optimizer_D, optimizer_G], [scheduler_D, scheduler_G]
