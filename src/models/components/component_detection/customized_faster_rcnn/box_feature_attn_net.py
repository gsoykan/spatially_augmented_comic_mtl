import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.transformer_net.positional_embedding import PositionalEncoding
from src.models.components.transformer_net.special_embedding_type import SpecialEmbeddingType


class BoxFeatureAttnNet(nn.Module):
    def __init__(self,
                 input_size=256,
                 hidden_size=256,
                 representation_dim=256,
                 v=0,
                 dropout=0.1,
                 use_positional_encoding: bool = True,
                 special_embedding_type: Optional[SpecialEmbeddingType] = None,
                 output_style: str = 'as_is'):
        super(BoxFeatureAttnNet, self).__init__()
        self.output_style = output_style
        self.use_positional_encoding = use_positional_encoding

        self.positional_encoding = nn.Identity()
        if use_positional_encoding:
            self.positional_encoder = PositionalEncoding(dim_model=input_size, dropout_p=dropout, max_len=7 * 7 * 3 + 5)

        self.special_embedding_type = special_embedding_type
        if special_embedding_type is SpecialEmbeddingType.CLS_SEP:
            self.special_embeddings = nn.Embedding(2, input_size)

        self.v = v
        self.hidden_size = hidden_size
        self.representation_dim = representation_dim
        self.input_size = input_size
        # Attention
        self.attention = nn.MultiheadAttention(input_size, num_heads=4, dropout=dropout)
        self.o_proj = nn.Linear(input_size, input_size)

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, representation_dim),
        )

    def forward(self,
                features: torch.Tensor):
        # TODO: @gsoykan - masking v4 gelebilir - transformer mask olarak kullanabiliriz
        #  elimizdekini ve epey ilginç olabilir...

        if self.special_embedding_type is SpecialEmbeddingType.CLS_SEP:
            # TODO: @gsoykan - face ve body ye özel ayrı cls tokenlar yapmak mantıklı mı?
            features = self._basic_add_cls_sep_embeddings_to_src(features)

        features = self.positional_encoder(features)  # [SeqLen, Batch, Dims]
        # Attention part
        attn_output, _ = self.attention(features, features, features)

        # TODO: @gsoykan - temp remove
        if self.output_style == 'dot_product':
            output = self._dot_prod(attn_output)
            return output

        output = self.o_proj(attn_output)
        x = features + self.dropout(output)
        x = self.norm1(x)
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        output = self.norm2(x)

        if self.output_style == 'return_cls':
            output = output[0]
        elif self.output_style == 'mean':
            output = torch.mean(output, dim=0)
        elif self.output_style == 'dot_product':
            self._dot_prod(output)
        return output

    def _dot_prod(self, output):
        encap_tokens, from_tokens, to_tokens = torch.split(output, output.shape[0] // 3, dim=0)
        from_mean = torch.mean(from_tokens, dim=0)  # [12, 256]
        to_mean = torch.mean(to_tokens, dim=0)
        output = torch.sum(from_mean * to_mean, dim=-1, keepdim=True)  # [12, 1]
        return output

    def _basic_add_cls_sep_embeddings_to_src(self,
                                             src: torch.Tensor,
                                             num_separator: int = 3) -> torch.Tensor:
        cls_tokens = torch.full((1, src.shape[1]), 0, dtype=torch.long, device=src.device)
        sep_tokens = torch.full((num_separator, src.shape[1]), 1, dtype=torch.long, device=src.device)
        cls_embeddings = self.special_embeddings(cls_tokens)
        sep_embeddings = self.special_embeddings(sep_tokens)
        sep_src = rearrange(src, "(n s) b f -> n s b f",
                            n=num_separator)
        final_src = [cls_embeddings]
        for i in range(num_separator):
            final_src.append(sep_src[i])
            final_src.append(sep_embeddings[i].view(1, *sep_embeddings[i].shape))
        return torch.cat(final_src, dim=0)
