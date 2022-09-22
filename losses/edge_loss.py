from typing import Callable

import torch
import torch.nn as nn
from kornia.color import rgb_to_grayscale
from kornia.filters import canny, laplacian, sobel
from torch.nn.functional import l1_loss


class EdgeLoss(nn.Module):
    """
    PyTorch module for Edge loss.
    """
    def __init__(self, operator: str='canny', loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=l1_loss):
        super(EdgeLoss, self).__init__()
        assert operator in {'canny', 'laplacian', 'sobel'}, 'operator must be one of {canny, laplacian, sobel}'
        self._loss_function = loss_function
        self._operator = operator

    def extract_edges(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            input_grayscale = rgb_to_grayscale(input_tensor)

            if self._operator == 'canny':
                return canny(input_grayscale)[0]
            elif self._operator == 'laplacian':
                kernel_size = input_grayscale.size()[-1] // 10
                if kernel_size % 2 == 0:
                    kernel_size += 1
                return laplacian(input_grayscale, kernel_size=kernel_size)
            elif self._operator == 'sobel':
                return sobel(input_grayscale)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prediction_edges = self.extract_edges(prediction)
            target_edges = self.extract_edges(target)

            return self._loss_function(prediction_edges, target_edges)
