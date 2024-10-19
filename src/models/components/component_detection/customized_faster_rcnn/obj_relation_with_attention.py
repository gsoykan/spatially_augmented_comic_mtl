import math

import torch
import torch.nn as nn


# source: https://github.dev/msracver/Relation-Networks-for-Object-Detection/blob/master/relation_rcnn/core/module.py
class ObjectRelationModuleWithAttention(nn.Module):
    def __init__(self,
                 fc_dim=16,
                 feat_dim=1024,
                 dim=(1024, 1024, 1024),
                 group=16,
                 index=1):
        super(ObjectRelationModuleWithAttention, self).__init__()
        self.fc_dim = fc_dim
        self.feat_dim = feat_dim
        self.dim = dim
        self.group = group
        self.index = index  # this shows after which fc it is applied to...

        self.dim_group = (dim[0] // group, dim[1] // group, dim[2] // group)
        self.query = nn.Linear(feat_dim, dim[0])
        self.key = nn.Linear(feat_dim, dim[1])
        # in original implementation value is identity and the linear layer for that is commented out
        # v_data = nongt_roi_feat
        # # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
        self.value = nn.Linear(feat_dim, dim[2]) # nn.Identity()
        self.pair_pos_fc1 = nn.Conv2d(self.dim_group[0], fc_dim, kernel_size=1, stride=1, padding=0)
        self.linear_out = nn.Conv2d(fc_dim * feat_dim, dim[2], kernel_size=1, groups=fc_dim)

    def forward(self, roi_feats, position_embeddings, non_gt_index=None):
        """ Attention module with vectorized version

                Args:
                    roi_feat: [num_rois, feat_dim]
                    position_embedding: [1, emb_dim, num_rois, nongt_dim]
                    non_gt_index:
                    fc_dim: should be same as group
                    feat_dim: dimension of roi_feat, should be same as dim[2]
                    dim: a 3-tuple of (query, key, output)
                    group:
                    index:

                Returns:
                    output: [num_rois, ovr_feat_dim, output_dim]
        """
        outputs = []
        for roi_feat, position_embedding in zip(roi_feats, position_embeddings):
            if non_gt_index is None:
                nongt_roi_feat = roi_feat
            else:
                nongt_roi_feat = roi_feat[non_gt_index]

            # position_embedding: [num_rois, nongt_dim, 64]
            position_embedding_reshape = torch.transpose(position_embedding, 0, 2)  # [64, num_rois, nongt_dim]
            position_embedding_reshape = torch.unsqueeze(position_embedding_reshape,
                                                         dim=0)  # [1, 64, num_rois, nongt_dim]

            # [1, emb_dim, num_rois, nongt_dim]
            # position_feat_1, [1, fc_dim, num_rois, nongt_dim]
            position_feat_1 = self.pair_pos_fc1(position_embedding_reshape)
            position_feat_1_relu = nn.functional.relu(position_feat_1)

            # aff_weight, [num_rois, fc_dim, nongt_dim, 1]
            aff_weight = position_feat_1_relu.permute(2, 1, 3, 0)
            # aff_weight, [num_rois, fc_dim, nongt_dim]
            aff_weight = aff_weight.reshape(aff_weight.size(0), aff_weight.size(1), -1)  # [M, 16, M] M=512

            # multi head
            assert self.dim[0] == self.dim[1], 'Matrix multiply requires same dimensions!'
            q_data = self.query(roi_feat)
            q_data_batch = q_data.reshape(-1, self.group, self.dim_group[0]).transpose(0, 1)
            k_data = self.key(nongt_roi_feat)
            k_data_batch = k_data.reshape(-1, self.group, self.dim_group[1]).transpose(0, 1)
            v_data = self.value(nongt_roi_feat)
            # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
            aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))  # [16, 512, 512]
            # aff_scale, [group, num_rois, nongt_dim]
            aff_scale = (1.0 / math.sqrt(float(self.dim_group[1]))) * aff
            aff_scale = aff_scale.permute(1, 0, 2)

            assert self.fc_dim == self.group, 'fc_dim != group'
            # weighted_aff, [num_rois, fc_dim, nongt_dim]
            weighted_aff = torch.log(aff_weight) + aff_scale
            aff_softmax = torch.softmax(weighted_aff, dim=2)
            aff_softmax_reshape = aff_softmax.view(aff_softmax.shape[0] * aff_softmax.shape[1], -1)
            # [num_rois * fc_dim, nongt_dim, feat_dim]
            output_t = torch.matmul(aff_softmax_reshape, v_data)

            # [num_rois, fc_dim * feat_dim, 1, 1]
            output_t = output_t.view(-1, self.fc_dim * self.feat_dim, 1, 1)
            # [num_rois, dim[2], 1, 1]
            linear_out = self.linear_out(output_t)
            # [num_rois, ovr_feat_dim, output_dim]
            output = linear_out.view(-1, self.dim[2])
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        return outputs

    @staticmethod
    def extract_position_matrix(proposals, non_gt_index=None):
        """ Extract position matrix

        Args:
            bbox: [num_boxes, 4]

        Returns:
            position_matrices: List of [num_boxes, nongt_dim, 4]
        """
        position_matrices = []
        for bbox in proposals:
            xmin, ymin, xmax, ymax = torch.split(bbox, split_size_or_sections=1, dim=1)
            # [num_boxes, 1]
            bbox_width = xmax - xmin + 1.
            bbox_height = ymax - ymin + 1.
            center_x = 0.5 * (xmin + xmax)
            center_y = 0.5 * (ymin + ymax)
            # [nongt_dim, 1]
            if non_gt_index is None:
                bbox_width_nongt = bbox_width
                bbox_height_nongt = bbox_height
                center_x_nongt = center_x
                center_y_nongt = center_y
            else:
                bbox_width_nongt = bbox_width[non_gt_index, :]
                bbox_height_nongt = bbox_height[non_gt_index, :]
                center_x_nongt = center_x[non_gt_index, :]
                center_y_nongt = center_y[non_gt_index, :]
            # [num_boxes, nongt_dim]
            delta_x = center_x - torch.transpose(center_x_nongt, 0, 1)
            delta_x = delta_x / bbox_width
            delta_x = torch.log(torch.clamp(torch.abs(delta_x), min=1e-3))
            delta_y = center_y - torch.transpose(center_y_nongt, 0, 1)
            delta_y = delta_y / bbox_height
            delta_y = torch.log(torch.clamp(torch.abs(delta_y), min=1e-3))
            delta_width = bbox_width / torch.transpose(bbox_width_nongt, 0, 1)
            delta_width = torch.log(delta_width)
            delta_height = bbox_height / torch.transpose(bbox_height_nongt, 0, 1)
            delta_height = torch.log(delta_height)
            concat_list = [delta_x, delta_y, delta_width, delta_height]
            for idx, sym in enumerate(concat_list):
                concat_list[idx] = torch.unsqueeze(sym, dim=2)
            position_matrix = torch.cat(concat_list, dim=2)
            position_matrices.append(position_matrix)
        return position_matrices

    @staticmethod
    def extract_position_embedding(position_matrices, feat_dim=64, wave_length=1000):
        embeddings = []
        for position_mat in position_matrices:
            # position_mat, [num_rois, nongt_dim, 4]
            feat_range = torch.arange(0, feat_dim / 8, device=position_mat.device)
            dim_mat = torch.pow(torch.full((1,), wave_length, device=position_mat.device), (8. / feat_dim) * feat_range)
            dim_mat = dim_mat.view((1, 1, 1, -1))
            position_mat = torch.unsqueeze(100.0 * position_mat, dim=3)
            div_mat = torch.div(position_mat, dim_mat)
            sin_mat = torch.sin(div_mat)
            cos_mat = torch.cos(div_mat)
            # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
            embedding = torch.cat((sin_mat, cos_mat), dim=3)
            # embedding, [num_rois, nongt_dim, feat_dim]
            embedding = embedding.view(embedding.shape[0], embedding.shape[1], feat_dim)
            embeddings.append(embedding)  # [N, N, 64]
        return embeddings
