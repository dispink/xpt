import torch
from torch import nn
import numpy as np


class InstanceNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if x.shape[-1] == 1:
            return x
        return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)


class Normalize(nn.Module):
    def __init__(self, mean, std, eps=1e-8):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def forward(self, x: torch.Tensor):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return torch.where(self.std == 0, 0, (x - self.mean) / (self.std + self.eps))


class LogTransform(nn.Module):
    def __init__(self, eps=1):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return torch.log(x + self.eps)
