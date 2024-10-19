import torch
import torch.nn as nn

from einops import rearrange

from src.models.components.transformer_net.positional_embedding import PositionalEncoding


# source: https://spotintelligence.com/2023/01/31/self-attention/
# https://github.com/sooftware/attentions/tree/master
class FaceToBodyAttentionFusion(nn.Module):
    def __init__(self,
                 single_input_dim,
                 output_dim,
                 use_positional_encoding: bool = True, ):
        super(FaceToBodyAttentionFusion, self).__init__()

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
        face_x = x[:, 0]
        # body_x = x[:, 1]

        queries = self.query(face_x).unsqueeze(1)  # [b, 1, o]
        keys = self.key(x)  # [b, 2, o]
        values = self.value(x)  # [b, 2, o]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values).squeeze(1)  # [B, output_dim]
        return weighted


if __name__ == '__main__':
    fake_face = torch.randn(4, 2048)
    fake_body = torch.randn(4, 2048)
    fusion_model = FaceToBodyAttentionFusion(2048, 256)
    result = fusion_model(fake_face, fake_body)
    print(result)
