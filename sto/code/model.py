import torch
from torch import nn
from torch import Tensor


class SimpleModel(nn.Module):

    def __init__(self, dim: int):
        super(SimpleModel, self).__init__()

        self.w = nn.Parameter(torch.randn(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w
