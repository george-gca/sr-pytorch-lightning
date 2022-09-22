from .edge_loss import EdgeLoss
from .flip import FLIP, FLIPLoss
from .losses import PSNR, VGG16, VGG19, GANLoss, TVLoss, VGGLoss
from .pencil_sketch import PencilSketchLoss

__all__ = [
    'EdgeLoss',
    'FLIP',
    'FLIPLoss',
    'GANLoss',
    'PencilSketchLoss',
    'PSNR',
    'TVLoss',
    'VGG16',
    'VGG19',
    'VGGLoss',
]
