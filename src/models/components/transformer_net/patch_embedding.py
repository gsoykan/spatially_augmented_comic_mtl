from typing import Optional

import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
import enum


class PatchProjectionMode(enum.Enum):
    Linear = 1
    Conv = 2


class PatchInputMode(enum.Enum):
    IMAGE = 'IMAGE'
    IMAGE_EMBEDDING = 'IMAGE_EMBEDDING'


# expected img size is 224 * 224
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: Optional[int] = 3,
                 patch_size: int = 16,
                 emb_size: Optional[int] = 256,
                 projection_mode: Optional[PatchProjectionMode] = PatchProjectionMode.Linear,
                 input_mode: PatchInputMode = PatchInputMode.IMAGE):
        self.patch_size = patch_size
        if isinstance(projection_mode, str):
            projection_mode = PatchProjectionMode(projection_mode)
        self.projection_mode = projection_mode

        if isinstance(input_mode, str):
            input_mode = PatchInputMode(input_mode)
        self.input_mode = input_mode

        super().__init__()
        if self.input_mode == PatchInputMode.IMAGE:
            if projection_mode == PatchProjectionMode.Linear:
                self.projection = nn.Sequential(
                    # break-down the image in s1 x s2 patches and flat then
                    Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
                              s1=patch_size,
                              s2=patch_size),
                    nn.Linear(patch_size * patch_size * in_channels, emb_size)
                )
            elif projection_mode == PatchProjectionMode.Conv:
                self.projection = nn.Sequential(
                    # using a conv layer instead of a linear one -> performance gains
                    nn.Conv2d(in_channels,
                              emb_size,
                              kernel_size=patch_size,
                              stride=patch_size),
                    Rearrange('b e (h) (w) -> b (h w) e'),
                )
        elif self.input_mode == PatchInputMode.IMAGE_EMBEDDING:
            # JUST REARRANGES IMAGE EMBEDDING
            if projection_mode == PatchProjectionMode.Linear:
                # panel_count, batch, embedding_dim
                self.projection = nn.Sequential(
                    # break-down the image embedding in s1 patches
                    Rearrange('p b (c s1) -> p c b s1',
                              s1=patch_size)
                )
            elif projection_mode == PatchProjectionMode.Conv:
                raise Exception('IMAGE_EMBEDDING MODE can not have CONV projection!')
        else:
            raise Exception('UNKNOWN PATCH EMBEDDING MODE')

    def forward(self, x: Tensor):
        return self.projection(x)  # [1, 196, 512], in image case,
        # in VGG feature case(feature dim=4096) [P, 16, B, 256]
