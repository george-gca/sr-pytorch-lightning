import torch
import torch.nn as nn
from kornia.color import rgb_to_grayscale
from kornia.enhance import invert
from kornia.filters.gaussian import gaussian_blur2d
from piq import psnr


class PencilSketchLoss(nn.Module):
    """
    PyTorch module for Pencil Sketch loss.
    This code is inspired by https://gitlab.com/eldrey/eesr-masters_project/-/blob/master/src/loss/pencil_sketch.py.
    """
    def __init__(self):
        super(PencilSketchLoss, self).__init__()

    def pencil_sketch(self, input: torch.Tensor, kernel_size: int = -1, sigma: float = 1., border_type: str = 'reflect') -> torch.Tensor:
        with torch.no_grad():
            if kernel_size == -1:
                kernel_size = input.size()[-1] // 10
                if kernel_size % 2 == 0:
                    kernel_size += 1

            grayscale = rgb_to_grayscale(input)
            inverted_grayscale = invert(grayscale)
            inverted_gaussian_blurred = gaussian_blur2d(inverted_grayscale, (kernel_size, kernel_size), (sigma, sigma), border_type)
            gaussian_blurred = invert(inverted_gaussian_blurred)
            ps = torch.div(grayscale, gaussian_blurred)
            ps[torch.isnan(ps)] = 0
            return ps.clamp(0, 1)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_pencil_sketch = self.pencil_sketch(prediction)
        target_pencil_sketch = self.pencil_sketch(target)
        return 100 - psnr(prediction_pencil_sketch, target_pencil_sketch)
