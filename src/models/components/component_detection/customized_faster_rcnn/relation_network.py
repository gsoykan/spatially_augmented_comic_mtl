from itertools import combinations
from typing import Optional, Tuple

import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from src.models.components.component_detection.customized_faster_rcnn.box_feature_attn_net import BoxFeatureAttnNet
from src.models.components.component_detection.customized_faster_rcnn.conv_rel_rep_network import \
    ConvRelationRepresentationNetwork
from src.models.components.component_detection.customized_faster_rcnn.fast_rcnn_conv import FastRCNNConv
from src.models.components.component_detection.customized_faster_rcnn.feature_rel_attn_net import FeatureRelationAttnNet
from src.models.components.component_detection.customized_faster_rcnn.rel_res_net import RelationResNet
from src.models.components.transformer_net.positional_embedding import PositionalEncoding

import torch.nn.functional as F

from src.models.components.transformer_net.special_embedding_type import SpecialEmbeddingType


class LambdaLayer(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def get_pairwise_features(features: torch.Tensor):
    comb = list(combinations(range(features.shape[1]), 2))
    pairwise_features = torch.cat(
        [features[:, i].unsqueeze(1) * features[:, j].unsqueeze(1) for i, j in comb],
        dim=2)  # Output size: (B, 1, C'* 3, H', W')
    return pairwise_features


def get_triplet_wise_features(features: torch.Tensor):
    comb = list(combinations(range(features.shape[1]), 3))
    triple_features = torch.cat(
        [features[:, i].unsqueeze(1) * features[:, j].unsqueeze(1) * features[:, k].unsqueeze(1) for i, j, k in comb],
        dim=2)  # Output size: (B, 1, C', H', W')
    return triple_features


def get_pair_and_triplet_wise_features(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pairwise_features = get_pairwise_features(features)
    triplet_wise_features = get_triplet_wise_features(features)
    return pairwise_features, triplet_wise_features


def make_relational_reasoning(x, score_mlp):
    # x has shape (batch_size, num_objects, feature_size)

    # Compute all pairs of object features
    num_objects = x.size(1)
    pairs = torch.cat([x.unsqueeze(1).expand(-1, num_objects, -1, -1),
                       x.unsqueeze(2).expand(-1, -1, num_objects, -1)],
                      dim=-1)  # shape (batch_size, num_objects, num_objects, feature_size*2)

    # Apply MLP to each pair of object features
    h = score_mlp(pairs)

    # Compute the final relation score for each pair of objects
    scores = h.sum(dim=-1)  # shape (batch_size, num_objects, num_objects)

    # Normalize the scores across object pairs using softmax
    scores = F.softmax(scores, dim=-1)

    # Compute the weighted sum of object features using the relation scores
    output = torch.bmm(scores, x)  # shape (batch_size, num_objects, feature_size)
    # NOTE: bu aslında bize niye uygun değil? çünkü biz pairlerle alakalı 1 - 0 ya da
    # nasıl ilişkileri var sorusunu sormuyoruz? belki implicitly soruyoruz.
    # ama paper da bütün objeleri birbiriyle çarpıştırıyor..
    # ne faydası olabilir, encapsulation box a daha fazla weight verebilir?
    # ama düşünce deneyimde softmax sonucu hep 0.33, 0.33, 0.33 olmalı
    return output


def merge_pair_and_triplet_wise_features(features: torch.Tensor):
    pairwise_features, triplet_wise_features = get_pair_and_triplet_wise_features(features)
    relation_features = torch.cat([pairwise_features, triplet_wise_features], dim=2)  # [B, 1, C(256) * 4, 7, 7]
    relation_features = rearrange(relation_features, "b d c h w -> b (d c) h w")
    return relation_features  # [B, C(256) * 4, 7, 7]


def merge_pair_and_triplet_wise_features_and_cat(features: torch.Tensor):
    pairwise_features, triplet_wise_features = get_pair_and_triplet_wise_features(features)
    relation_features = torch.cat([pairwise_features, triplet_wise_features], dim=2)  # [B, 1, C(256) * 4, 7, 7]
    relation_features = rearrange(relation_features, "b d (n c) h w -> b n (d c) h w", n=4)  # [B, 4, 256, 7, 7]
    relation_features = torch.cat([features, relation_features], dim=1)  # [B, 7, 256, 7, 7]
    return relation_features


def process_encapsulation_and_others_separately_as_dot_product(x, others_processor):
    others = x[:, 1:].view(x.shape[0], -1)  # [B, 2 * representation]
    others = others_processor(others)  # [B, representation]
    dot_product = torch.bmm(x[:, 0].unsqueeze(1), others.unsqueeze(2)).squeeze()  # [B]
    modified_encapsulation_features = x[:, 0] * dot_product.view(-1, 1)  # [B, R]
    return modified_encapsulation_features


def process_face_encap_and_speech_encap(x, x_processor):  # x -> B, C, 256, 7, 7
    x_flat = torch.flatten(x, start_dim=3)
    encapsu_flat = x_flat[:, 0]
    speech_flat = x_flat[:, 1]
    face_flat = x_flat[:, 2]
    encapsu_speech_dot = torch.bmm(encapsu_flat, speech_flat.transpose(1, 2)).sum(-1)  # [B, 256]
    encapsu_face_dot = torch.bmm(encapsu_flat, face_flat.transpose(1, 2)).sum(-1)  # [B, 256]
    pair_res = torch.cat((encapsu_speech_dot, encapsu_face_dot), dim=1)  # [B, 512]
    x = x_processor(x)  # [B, 3 * representation]
    all_res = torch.cat((x, pair_res), dim=1)  # [B, 3 * rep + 512]
    return all_res


class RelationNetwork(nn.Module):
    def __init__(self,
                 in_channels,
                 representation_size,
                 spatial_feature_size=5,
                 first_layer_type='linear',
                 feat_embedding_type: Optional[str] = None,
                 encapsulation_box_masks_strategy: Optional[str] = None,
                 box_head_output_strategy: Optional[str] = None,
                 use_encapsulation_edge_maps: bool = False
                 ):
        """
        resolution = box_roi_pool.output_size[0]
            representation_size = 128
            stacked_feature_size = 3  # 1 for each, speech + char + context
            relation_network = RelationNetwork(
                stacked_feature_size * out_channels * resolution ** 2,
                representation_size)
        Args:
            in_channels ():
            representation_size ():
            spatial_feature_size ():
            first_layer_type ():  can be linear, conv3d, depthwise_conv, conv_rel_rep, fast_rcnn_conv
            feat_embedding_type (): 'dynamic', 'positional' ...
            todo: try static embeddings like positional embeddings
            nereye ekleneceği - handi dimensionlarla ekleneceği, linear ve conv lar için
            nasıl stratejiler izleneceği önemli...
            encapsulation_box_masks_strategy (): 'element_wise', 'cat_and_conv'
            cat_and_conv: ('256' lık encapsu feature lara direk ekleyip sonra conv ile channel ı 256 ya tekrar indirmek)
            box_head_output_strategy (): 'cat_and_linear'
        """
        super(RelationNetwork, self).__init__()
        self.use_encapsulation_edge_maps = use_encapsulation_edge_maps
        if self.use_encapsulation_edge_maps:
            self.encapsulation_edge_map_processor = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=14, stride=8)
            self.edge_map_merger = nn.Sequential(
                Rearrange('b c d h w -> (b c) d h w'),
                nn.Conv2d(in_channels=257, out_channels=256, kernel_size=1, stride=1),
                Rearrange('(b c) d h w -> b c d h w', c=3)
            )

        self.encapsulation_box_masks_strategy = encapsulation_box_masks_strategy
        if self.encapsulation_box_masks_strategy == 'cat_and_conv':
            self.encapsulation_mask_conv = nn.Conv2d(257, 256, kernel_size=(3, 3), padding=1, stride=1)

        self.feat_embedding_type = feat_embedding_type
        self.preprocess_layer = nn.Identity()
        if self.feat_embedding_type == 'dynamic':
            self.embedding_dim = 256
            self.feat_embedding = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_dim)
        elif self.feat_embedding_type == 'positional':
            self.embedding_dim = 256
            self.feat_embedding = PositionalEncoding(dim_model=self.embedding_dim, dropout_p=0, max_len=100)
        else:
            self.embedding_dim = 0
            self.feat_embedding = nn.Identity()

        self.first_layer_type = first_layer_type
        if self.first_layer_type == 'linear':
            self.preprocess_layer = Rearrange('b c d h w -> b (c d h w)')
            self.representation_model = nn.Sequential(
                nn.Linear(in_channels + self.embedding_dim, representation_size),
                nn.ReLU(),
                nn.Linear(representation_size, representation_size),  # fc7
                nn.ReLU(),
            )
        elif self.first_layer_type == 'conv3d':
            # TODO: @gsoykan - you can try to parameterize for this...
            self.representation_model = nn.Sequential(
                nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(3, 7, 7), stride=2, padding=(1, 0, 0)),
                Rearrange('b c d h w -> b (c d h w)'),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'depthwise_conv3d':
            self.representation_model = nn.Sequential(
                nn.Conv3d(in_channels=3, out_channels=3, groups=3, kernel_size=(3, 7, 7), stride=2,
                          padding=(1, 0, 0)),
                nn.ReLU(),
                Rearrange('b c d x y -> b c (d x y)'),
                nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
                Rearrange('b r d -> b (r d)'),
                nn.ReLU(),
                nn.Linear(representation_size, representation_size),  # fc7
                nn.ReLU(),
            )
        elif self.first_layer_type == 'pair_triple_wise_conv2d':
            self.representation_model = nn.Sequential(
                LambdaLayer(func=merge_pair_and_triplet_wise_features),  # [B, 4 * 256, 7, 7],
                nn.Conv2d(4 * 256, 256, kernel_size=(3, 3), padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(256, 256, groups=256, kernel_size=(3, 3), padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=(2, 2), padding=0, stride=1),
                Rearrange('b r x y -> b (r x y)', x=1, y=1),
            )
        elif self.first_layer_type == 'conv_rel_rep':
            self.representation_model = ConvRelationRepresentationNetwork(input_channels=256,
                                                                          feature_dim=64,
                                                                          representation_size=representation_size)
        elif self.first_layer_type == 'rel_res_net':
            self.representation_model = RelationResNet(input_channels=256,
                                                       representation_size=representation_size)
        elif self.first_layer_type == 'fast_rcnn_conv_v1':
            # burada opsiyonlar ne
            # 1) channel dim üzerinden stacklemek
            self.preprocess_layer = Rearrange('b c d h w -> b (c d) h w')
            self.representation_model = FastRCNNConv(
                (256 * 3, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v2':
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            self.representation_model = nn.Sequential(FastRCNNConv(
                (256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d
            ),
                Rearrange('(b c) d -> b (c d)', c=3),
                nn.Linear(3 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v2_regularized':
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            self.representation_model = nn.Sequential(FastRCNNConv(
                (256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d
            ),
                Rearrange('(b c) d -> b (c d)', c=3),
                nn.Linear(3 * representation_size, representation_size),
                nn.ReLU(),
                nn.Dropout(p=0.5)  # Adding dropout regularization with a dropout rate of 0.5
            )
        elif self.first_layer_type == 'pair_triple_wise_and_fast_rcnn_conv_v2':
            self.representation_model = nn.Sequential(
                LambdaLayer(func=merge_pair_and_triplet_wise_features),  # [B, 4 * 256, 7, 7],
                Rearrange('b (c d) h w -> (b c) d h w', c=4),
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size],
                             norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b (c d)', c=4),
                nn.Linear(4 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'pair_triple_wise_cat_and_fast_rcnn_conv_v2':
            self.representation_model = nn.Sequential(
                LambdaLayer(func=merge_pair_and_triplet_wise_features_and_cat),  # [B, 7, 256, 7, 7],
                Rearrange('b c d h w -> (b c) d h w', c=7),
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size],
                             norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b (c d)', c=7),
                nn.Linear(7 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v3':
            # 3) yine hepsini ayrı ayrı işleyip daha sonra pair - triple olarak eşleyip kullanmak
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            self.representation_model = nn.Sequential(FastRCNNConv(
                (256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d
            ),
                Rearrange('(b c) d -> b c d', c=3),
                LambdaLayer(func=get_pair_and_triplet_wise_features),
                LambdaLayer(func=lambda x: torch.cat((x[0], x[1]), dim=2).squeeze(dim=1)),
                nn.Linear(4 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v4':
            # burada [b c d] şeklinde alıyoruz ve sonra box_head output ile cat layıp kullancaz
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            self.representation_model = nn.Sequential(
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b c d', c=3),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v5':
            # 5) hepsini ayrı ayrı işleyip sonra face ve speech ten feature çıkartıp
            # encapsu ile çarpıştırmak (dot_product anlamında)
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            self.others_processor = nn.Sequential(
                nn.Linear(2 * representation_size, representation_size),
                nn.ReLU(),
            )
            self.representation_model = nn.Sequential(FastRCNNConv(
                (256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d
            ),
                Rearrange('(b c) d -> b c d', c=3),
                LambdaLayer(
                    func=lambda x: process_encapsulation_and_others_separately_as_dot_product(x, self.others_processor))
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v6':
            # 7) roi_featureları (face ve speech) ten featureları bmm ile kullanmak...
            self.preprocess_layer = nn.Identity()  # Rearrange('b c d h w -> (b c) d h w')
            # process_face_encap_and_speech_encap
            self.core_representation_model = nn.Sequential(
                Rearrange('b c d h w -> (b c) d h w'),
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b (c d)', c=3))
            self.representation_model = nn.Sequential(
                LambdaLayer(func=lambda x: process_face_encap_and_speech_encap(x, self.core_representation_model)),
                nn.Linear(3 * representation_size + 512, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v7':
            # a simple neural network module for relational reasoning
            # paperındaki relation networkten etkilenerek kullanıyoruz...
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            # input for this is pairs...
            self.relation_score_model = nn.Sequential(
                nn.Linear(2 * representation_size, representation_size),
                nn.ReLU(),
                nn.Linear(representation_size, representation_size),
                nn.ReLU(),
                nn.Linear(representation_size, 1)
            )
            self.representation_model = nn.Sequential(FastRCNNConv(
                (256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d
            ),
                Rearrange('(b c) d -> b c d', c=3),
                LambdaLayer(func=lambda x: make_relational_reasoning(x, self.relation_score_model)),
                Rearrange('b c d -> b (c d)', c=3),
                nn.Linear(3 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v8_mha':
            # a simple neural network module for relational reasoning
            # paperındaki relation networkten etkilenerek kullanıyoruz...
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            # input for this is pairs...
            self.representation_model = nn.Sequential(
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b c d', c=3),
                FeatureRelationAttnNet(v=0),
                Rearrange('b c d -> b (c d)', c=3),
                nn.Linear(3 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v8_mha_1':
            # a simple neural network module for relational reasoning
            # paperındaki relation networkten etkilenerek kullanıyoruz...
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            # input for this is pairs...
            self.representation_model = nn.Sequential(
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b c d', c=3),
                FeatureRelationAttnNet(v=1),  # just linear o_proj
                Rearrange('b c d -> b (c d)', c=3),
                nn.Linear(3 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v8_mha_2':
            # a simple neural network module for relational reasoning
            # paperındaki relation networkten etkilenerek kullanıyoruz...
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            # input for this is pairs...
            self.representation_model = nn.Sequential(
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b c d', c=3),
                FeatureRelationAttnNet(v=2),  # linear o_proj + residual + norm
                Rearrange('b c d -> b (c d)', c=3),
                nn.Linear(3 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v8_mha_3':
            # a simple neural network module for relational reasoning
            # paperındaki relation networkten etkilenerek kullanıyoruz...
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            # input for this is pairs...
            self.representation_model = nn.Sequential(
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b c d', c=3),
                FeatureRelationAttnNet(v=3),  # linear o_proj + residual + norm + two_layer mlp
                Rearrange('b c d -> b (c d)', c=3),
                nn.Linear(3 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v8_mha_4':
            # speech_face_relation - face_speech_relation
            self.preprocess_layer = Rearrange('b c d h w -> (b c) d h w')
            self.representation_model = nn.Sequential(
                FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d),
                Rearrange('(b c) d -> b c d', c=3),
                FeatureRelationAttnNet(v=4),  # linear o_proj + residual + norm + two_layer mlp
                nn.Linear(representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v8_mha_box':
            # bu en ambitious fikirlerden bir tanesi
            # box_featureların spatial dimensionları üzerine MHA
            self.preprocess_layer = Rearrange('b c d h w -> (c h w) b d')
            self.representation_model = nn.Sequential(
                # FastRCNNConv((256, 7, 7), [256, 256, 256, 256], [representation_size], norm_layer=nn.BatchNorm2d),
                BoxFeatureAttnNet(v=0),  # linear o_proj + residual + norm + two_layer mlp
                Rearrange('l b d -> b (l d)'),
                nn.Linear(in_channels, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'mha_box_dot_product':
            # bu en ambitious fikirlerden bir tanesi
            # box_featureların spatial dimensionları üzerine MHA
            self.preprocess_layer = Rearrange('b c d h w -> (c h w) b d')
            self.representation_model = nn.Sequential(
                BoxFeatureAttnNet(v=0, output_style='dot_product'),  # linear o_proj + residual + norm + two_layer mlp
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v8_mha_box_2':
            # bu en ambitious fikirlerden bir tanesi
            # box_featureların spatial dimensionları üzerine MHA
            self.preprocess_layer = Rearrange('b c d h w -> (c h w) b d')
            self.representation_model = nn.Sequential(
                BoxFeatureAttnNet(v=0,
                                  use_positional_encoding=True,
                                  special_embedding_type=SpecialEmbeddingType.CLS_SEP,
                                  output_style='return_cls'),  # returns [B, 256]
                nn.Linear(256, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'fast_rcnn_conv_v8_mha_box_3':
            # bu en ambitious fikirlerden bir tanesi
            # box_featureların spatial dimensionları üzerine MHA
            self.preprocess_layer = Rearrange('b c d h w -> (c h w) b d')
            self.representation_model = nn.Sequential(
                BoxFeatureAttnNet(v=0,
                                  use_positional_encoding=True,
                                  special_embedding_type=SpecialEmbeddingType.CLS_SEP,
                                  output_style='mean'),  # returns [B, 256]
                nn.Linear(256, representation_size),
                nn.ReLU(),
            )
        elif self.first_layer_type == 'identity':
            # 4) representation layer hiç kullanmayıp sadece box_head output'unu alıp pair-triple olarak kullanmak...
            # bunun inputu box_head_output_strategy=linear outputu..
            self.representation_model = nn.Identity()

        self.box_head_output_strategy = box_head_output_strategy
        self.box_head_output_processor = None
        self.box_head_output_merge_layer = LambdaLayer(func=lambda x: x[0] + x[1])  # basic sum...
        if self.box_head_output_strategy == 'cat_and_linear':
            self.box_head_output_processor = nn.Sequential(
                Rearrange('b c f -> b (c f)'),
                nn.Linear(3 * 1024, representation_size),
                nn.ReLU(),
            )
        elif self.box_head_output_strategy == 'identity_and_sum_sum':
            assert representation_size == 1024, 'representation size should be 1024 for this output strategy'
            self.box_head_output_processor = Reduce('b c f -> b f', 'sum')
        elif self.box_head_output_strategy == 'linear_and_sum_sum':
            self.box_head_output_processor = nn.Sequential(
                nn.Linear(1024, representation_size),
                nn.ReLU(),
                Reduce('b c f -> b f', 'sum'),
            )
        elif self.box_head_output_strategy == 'linear_pair_triple_as_x':
            self.box_head_output_merge_layer = LambdaLayer(func=lambda x: x[1])  # ignore x just use box_head_output
            self.box_head_output_processor = nn.Sequential(
                nn.Linear(1024, representation_size),
                nn.ReLU(),  # [B, 3, 128]
                LambdaLayer(func=get_pair_and_triplet_wise_features),
                LambdaLayer(func=lambda x: torch.cat((x[0], x[1]), dim=2).squeeze(dim=1)),
                nn.Linear(4 * representation_size, representation_size),
                nn.ReLU(),
            )
        elif self.box_head_output_strategy == 'merge_cat_and_linear_with_x':
            self.box_head_output_processor = nn.Sequential(
                nn.Linear(1024, representation_size),
                nn.ReLU(),  # [B, 3, 128]
            )
            self.box_head_output_merge_layer = nn.Sequential(
                LambdaLayer(func=lambda x: torch.cat((x[0], x[1]), dim=2)),  # [b c 2*d]
                Rearrange('b c f -> b (c f)'),
                nn.Linear(2 * 3 * representation_size, representation_size),
                nn.ReLU(),
            )

        if 'regularized' in self.first_layer_type:
            self.out = nn.Sequential(
                nn.Linear(representation_size + spatial_feature_size, 128),  # Reducing the dimension to 128
                nn.ReLU(),
                nn.Dropout(p=0.3),  # Adding dropout regularization with a dropout rate of 0.3
                nn.Linear(128, 1)
            )
        else:
            self.out = nn.Linear(representation_size + spatial_feature_size, 1)

    def apply_encapsulation_box_masking(self,
                                        x: torch.Tensor,
                                        encapsulation_box_masks: torch.Tensor) -> torch.Tensor:
        if self.encapsulation_box_masks_strategy == 'cat_and_conv':
            encapsulation_and_mask = torch.cat([x[:, 0], encapsulation_box_masks.unsqueeze(dim=1)], dim=1)
            x[:, 0] = self.encapsulation_mask_conv(encapsulation_and_mask)
        else:
            # fallback is element_wise
            x[:, 0] = x[:, 0] * encapsulation_box_masks.unsqueeze(dim=1)
        return x

    # TODO: @gsoykan - regularization kullanmak iyi fikir olabilir?
    def forward(self,
                x,  # [B, 3, 256, 7, 7]
                spatial_features,
                to_labels: Optional[torch.Tensor] = None,
                encapsulation_box_masks: Optional[torch.Tensor] = None,  # [B, 7, 7]
                use_spatial: bool = True,
                box_head: Optional[nn.Module] = None,
                encapsulation_edge_maps: Optional[torch.Tensor] = None):

        box_head_output = None
        if box_head is not None and self.box_head_output_processor is not None:
            box_head_output = rearrange(x.clone(), "b c d h w -> (b c) d h w")
            box_head_output = box_head(box_head_output)
            box_head_output = rearrange(box_head_output, "(b c) f -> b c f", c=3)
            box_head_output = self.box_head_output_processor(box_head_output)

        if encapsulation_box_masks is not None:
            x = self.apply_encapsulation_box_masking(x, encapsulation_box_masks)

        if encapsulation_edge_maps is not None:
            e_m = self.encapsulation_edge_map_processor(encapsulation_edge_maps.unsqueeze(1))  # [B, 3, 7, 7]
            x = self.edge_map_merger(torch.cat([x, e_m.unsqueeze(2)], dim=2))  # [B, 3, 257(256), 7, 7]

        x = self.preprocess_layer(x)

        if not isinstance(self.feat_embedding, nn.Identity):
            # TODO: @gsoykan - this only works with linear layer r n
            # we substract 2 because char - face have 1, 2 as labels but emb. starts from 0, so it will 0 and -1
            if isinstance(self.feat_embedding, PositionalEncoding):
                to_label_features = self.feat_embedding.pos_encoding[to_labels - 2].squeeze(dim=1)
                x = torch.cat([x, to_label_features], dim=-1)
            else:
                to_label_features = self.feat_embedding(to_labels - 1)
                x = torch.cat([x, to_label_features], dim=-1)

        x = self.representation_model(x)
        if 'dot_product' in self.first_layer_type:
            return x

        if box_head_output is not None:
            x = self.box_head_output_merge_layer((x, box_head_output))

        if use_spatial:
            x = torch.cat([x, spatial_features], dim=1)

        x = self.out(x)
        return x
