"""Remove the decoder part of the model, and add fc layer to the encoder part.

It's copied from model_mae.py. Except the added head, the kept codes should be identical to it.
"""
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from util.patch_embed import PatchEmbed
from util.pos_embed import get_1d_sincos_pos_embed

class SpectrumRegressor(nn.Module):
    """ Masked Autoencoder without the masking part for downstream task:
        regression of CaCO3 and TOC.
    """
    # modify img_size to spe_size (2048)
    def __init__(self, spe_size=2048, patch_size=16,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        """
        spe_size: expected spectrum size (may be padded if inputting spectrum from other machines)
        patch_size: patch size for patch embedding
        embed_dim: dimension of embedding for transformer
        depth: number of blocks for transformer
        num_heads: number of multi-heads for transformer
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(spe_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # New head for regression: CaCO3 and TOC
        # output is fixed in 0-1
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.Sigmoid()
            )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear, nn.LayerNorm, nn.Conv1d
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # transform the cls to logits
        # (N, 1, 2)
        pred = self.fc(x[:, 0]) * 100  # scale to 0-100, relevent to weighting percent unit
        
        return pred
    
def mae_vit_base_patch16_dec512d8b(pretrained: bool, **kwargs):
    model = SpectrumRegressor(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if pretrained:
        # adopt pretrained model's weights to the new model
        pretrained_state = torch.load('models/mae_vit_base_patch16_l-coslr_1e-05_20231229.pth')
        model_state = model.state_dict()
        compiled_state = model_state.copy()

        # update the weights that can be found in the pretrained model
        for k, v in pretrained_state.items():
            if k in model_state:
                compiled_state[k] = v

        model.load_state_dict(compiled_state)

    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks