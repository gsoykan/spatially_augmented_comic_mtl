import math

import torch
import torch.nn as nn

from src.models.components.transformer_net.positional_embedding import PositionalEncoding


class FeatureRelationAttnNet(nn.Module):
    def __init__(self,
                 input_size=128,
                 hidden_size=128,
                 representation_dim=128,
                 v=2,
                 dropout=0.1,
                 use_positional_encoding: bool = True):
        super(FeatureRelationAttnNet, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        self.positional_encoding = nn.Identity()
        if use_positional_encoding:
            self.positional_encoder = PositionalEncoding(dim_model=input_size, dropout_p=dropout, max_len=3)

        self.v = v
        self.hidden_size = hidden_size
        self.representation_dim = representation_dim
        self.input_size = input_size
        # Attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=dropout)
        self.o_proj = nn.Identity()
        if v in [1, 2]:
            self.o_proj = nn.Linear(hidden_size, hidden_size)
        if v == 2:
            self.norm1 = nn.LayerNorm(input_size)
            self.dropout = nn.Dropout(dropout)
        if v == 3:
            self.norm1 = nn.LayerNorm(input_size)
            self.norm2 = nn.LayerNorm(input_size)
            self.dropout = nn.Dropout(dropout)
            self.linear_net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, representation_dim),
            )

    def forward(self, features):
        # TODO: @gsoykan - i think we can play better with attn_output
        features_permuted = features.permute(1, 0, 2)  # [SeqLen, Batch, Dims]
        features_permuted = self.positional_encoder(features_permuted)
        attn_output, _ = self.attention(features_permuted, features_permuted, features_permuted)
        attn_output = attn_output.permute(1, 0, 2)  # [Batch, SeqLen, Dims]
        output = self.o_proj(attn_output)
        if self.v == 4:
            speech_face_relation = attn_output[:, 1]
            face_speech_relation = attn_output[:, 2]
            output = speech_face_relation - face_speech_relation
        elif self.v == 2:
            x = features + self.dropout(output)
            output = self.norm1(x)
        elif self.v == 3:
            x = features + self.dropout(output)
            x = self.norm1(x)
            # MLP part
            linear_out = self.linear_net(x)
            x = x + self.dropout(linear_out)
            output = self.norm2(x)
        return output

    # def forward(self, features):
    #     box_features, speech_features, face_features = features[:, 0], features[:, 1], features[:, 2]
    #     batch_size = speech_features.shape[0]
    #
    #     # Encode features
    #     speech_features_encoded = self.encoder(speech_features).view(batch_size, -1, self.hidden_size)
    #     face_features_encoded = self.encoder(face_features).view(batch_size, -1, self.hidden_size)
    #     box_features_encoded = self.encoder(box_features).view(batch_size, -1, self.hidden_size)
    #
    #     # Compute pairwise relations
    #     speech_face_relation = \
    #         self.attention(speech_features_encoded.transpose(0, 1), face_features_encoded.transpose(0, 1),
    #                        box_features_encoded.transpose(0, 1))[0].transpose(0, 1)
    #     face_speech_relation = \
    #         self.attention(face_features_encoded.transpose(0, 1), speech_features_encoded.transpose(0, 1),
    #                        box_features_encoded.transpose(0, 1))[0].transpose(0, 1)
    #
    #     # Aggregate pairwise relations
    #     speech_face_relation = speech_face_relation.mean(dim=1)
    #     face_speech_relation = face_speech_relation.mean(dim=1)
    #
    #     # Decode relations
    #     output = self.decoder(speech_face_relation - face_speech_relation)
    #
    #     return output


"""
Subtracting `speech_face_relation` and `face_speech_relation` is 
a way of emphasizing the directionality of the relationship between the speech and face features. 

In particular, `speech_face_relation` captures the relationship between the speech and face features, with the speech
 features as queries and the face features as keys and values. On the other hand, `face_speech_relation` captures the 
 relationship between the face and speech features, with the face features as queries and the speech features as keys 
 and values.

By subtracting `speech_face_relation` from `face_speech_relation`, we are effectively emphasizing the directionality of 
the relationship from the face to the speech features, since the former has the face as queries and the speech as keys,
 while the latter has the speech as queries and the face as keys. This can be useful in tasks where the relationship
between the face and speech features is asymmetric, such as in identifying which speech bubbles belong to 
which character.
"""
