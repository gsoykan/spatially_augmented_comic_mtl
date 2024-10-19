import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.transformer_net.positional_embedding import PositionalEncoding


class CoefficientSumModule(nn.Module):
    def __init__(self,
                 input_dim,
                 use_positional_encoding: bool = True,):
        super(CoefficientSumModule, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, 2))
        nn.init.xavier_uniform_(self.weights)

        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding = nn.Identity()
        if use_positional_encoding:
            self.positional_encoder = PositionalEncoding(dim_model=input_dim,
                                                         dropout_p=0,
                                                         max_len=2)

    def forward(self, feature_all_face, feature_all_body):
        # Calculate weighted sum of face and body features
        weights = torch.softmax(self.weights, dim=1)

        if self.use_positional_encoding:
            x = torch.stack([feature_all_face, feature_all_body], dim=0)
            x = self.positional_encoder(x)  # [SeqLen, Batch, Dims]
            feature_all_face = x[0]
            feature_all_body = x[1]

        weighted_face = feature_all_face * weights[0, 0]
        weighted_body = feature_all_body * weights[0, 1]
        feature_fused = weighted_face + weighted_body
        return feature_fused
