"""Harmonize spectrum size to 1024
Different XRF machine may generate different spectrum size.
This class is to harmonize the spectrum size to fit later workflow.

The transform is inspired by NiN model:
https://d2l.ai/chapter_convolutional-modern/nin.html
"""

# TO BE DEVELOPED...
# The issue of inconsistent spectrum size between pred (transformed) 
# and target (original) is not solved yet.
# The inverse transform needs to be implemented in more aspects.


import torch
from torch import nn

class HarmonizeSpectrum(nn.Module):
    def __init__(self, h_size:int=1024):
        """
        any original spectrum size will be dealt with automatically
        no need to specify the original spectrum size
        h_size: spectrum size after harmonization
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(1, h_size, kernel_size=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    
    def forward(self, x):
        """
        x: (N, spe_size)
        output: (N, h_size)
        """
        x = x.unsqueeze(1) # (N, 1, spe_size)
        x = self.proj(x) # (N, h_size, spe_size) -> (N, h_size, 1) -> (N, h_size)
        return x

class Inverse_HarmonizeSpectrum(nn.Module):
    def __init__(self, spe_size):
        """
        spe_size: original spectrum size
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(1, spe_size, kernel_size=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    
    def forward(self, x):
        """
        x: (N, h_size)
        output: (N, spe_size)
        """
        x = x.unsqueeze(1) # (N, 1, spe_size)
        x = self.proj(x) # (N, h_size, spe_size) -> (N, h_size, 1) -> (N, h_size)
        return x