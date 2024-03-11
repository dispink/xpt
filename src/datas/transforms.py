import torch
import numpy as np


def standardize_numpy(spe):
    """
    spe: numpy array of shape (n_channels)
    Standardize the spectrum to have zero mean and unit variance.
    """
    mean = np.mean(spe, axis=0)
    std = np.std(spe, axis=0)
    spe = (spe - mean) / std
    return spe


def log_transform_numpy(spe):
    """
    spe: numpy array of shape (n_channels)
    Apply log transform to the spectrum.
    Add 1 to avoid log(0).
    """
    spe = np.log(spe+1)
    return spe


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return std * x + mean


def log_transform(x, eps=1):
    return torch.log(x + eps)


def exp_transform(x, eps=1):
    return torch.exp(x) - eps


class NormalizeTransform():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def apply(self, x: torch.Tensor):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return normalize(x, self.mean, self.std)

    def reverse(self, x: torch.Tensor):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return denormalize(x, self.mean, self.std)


class LogTransform():
    def __init__(self, eps=1):
        self.eps = eps

    def apply(self, x: torch.Tensor):
        return log_transform(x, self.eps)

    def reverse(self, x: torch.Tensor):
        return denormalize(x, self.eps)
