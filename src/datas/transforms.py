import torch
from torch import nn
import numpy as np


class InstanceNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return (x - self.mean) / self.std


class LogTransform(nn.Module):
    def __init__(self, eps=1):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return torch.log(x + self.eps)
