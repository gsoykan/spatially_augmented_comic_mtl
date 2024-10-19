import torch
import torch.nn as nn


class WeightedSumModule(nn.Module):
    def __init__(self, input_dim):
        super(WeightedSumModule, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_dim, 2))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feature_all_face, feature_all_body):
        # Calculate weighted sum of face and body features
        weights = torch.softmax(self.weights, dim=1)
        weighted_face = feature_all_face * weights[:, 0]
        weighted_body = feature_all_body * weights[:, 1]
        feature_fused = weighted_face + weighted_body
        return feature_fused
