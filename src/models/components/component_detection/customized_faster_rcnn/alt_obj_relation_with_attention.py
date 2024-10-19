import math

import torch
import torch.nn as nn


class AltObjectRelationModule(nn.Module):
    def __init__(self, in_channels=1024, num_heads=4, hidden_channels=1024):
        """
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
        Args:
            in_channels ():
            num_heads ():
            hidden_channels ():
        """
        super().__init__()
        self.query = nn.Linear(in_channels, hidden_channels, bias=False)
        self.key = nn.Linear(in_channels, hidden_channels, bias=False)
        self.value = nn.Linear(in_channels, hidden_channels, bias=False)
        self.multihead_attn = nn.MultiheadAttention(hidden_channels, num_heads)

    def forward(self, roi_feats):
        x = torch.stack(roi_feats, dim=1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output, _ = self.multihead_attn(q, k, v)
        return attn_output.view(-1, attn_output.shape[-1])
