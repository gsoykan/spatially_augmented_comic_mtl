import torch
import torch.nn as nn
import torch.nn.functional as F

# source: https://spotintelligence.com/2023/01/31/self-attention/
# https://github.com/sooftware/attentions/tree/master
from einops import rearrange

from src.models.components.transformer_net.positional_embedding import PositionalEncoding


class SelfAttentionFusion(nn.Module):
    def __init__(self,
                 single_input_dim,
                 output_dim,
                 use_positional_encoding: bool = True, ):
        super(SelfAttentionFusion, self).__init__()

        self.positional_encoding = nn.Identity()
        if use_positional_encoding:
            self.positional_encoder = PositionalEncoding(dim_model=single_input_dim,
                                                         dropout_p=0,
                                                         max_len=2)

        self.input_dim = single_input_dim
        self.query = nn.Linear(single_input_dim, output_dim)
        self.key = nn.Linear(single_input_dim, output_dim)
        self.value = nn.Identity()  # nn.Linear(single_input_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, face_features, body_features):
        x = torch.stack([face_features, body_features], dim=0)
        x = self.positional_encoder(x)  # [SeqLen, Batch, Dims]
        x = rearrange(x, "s b f -> b s f")

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)  # [B, 2, output_dim]
        fused_features = rearrange(weighted, "b s f -> b (s f)")  # [B, 2 * output_dim]
        return fused_features


if __name__ == '__main__':
    fake_face = torch.randn(4, 2048)
    fake_body = torch.randn(4, 2048)
    fusion_model = SelfAttentionFusion(2048, 256)
    result = fusion_model(fake_face, fake_body)
    print(result)
