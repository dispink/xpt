""" Spectrum to Patch Embedding using Conv1d

A convolution based approach to patchifying a 1D spectrum w/ embedding projection.

Codes are largely simplified and modified from:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision
  * https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
"""

import torch
from torch import nn

class PatchEmbed(nn.Module):
    """ 1D spectrum to Patch Embedding
    x: (N, spe_size)
    output: (N, num_patches, embed_dim)
    """

    def __init__(self, spe_size, patch_size, embed_dim, bias: bool = True):
        super().__init__()
        assert spe_size % patch_size == 0, f"Patch size {patch_size} should divide spectrum size {spe_size}."
        
        self.patch_size = patch_size
        self.num_patches = spe_size // self.patch_size
        
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias) 

    def forward(self, x):
        x = x.unsqueeze(1) # (N, 1, spe_size)
        x = self.proj(x) # (N, embed_dim, num_patches)
        x = x.transpose(1, 2) # (N, num_patches, embed_dim)
        return x
