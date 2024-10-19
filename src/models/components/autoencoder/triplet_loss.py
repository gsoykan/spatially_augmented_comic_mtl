"""This code was imported from tbmoon's 'facenet' repository:
   https://github.com/tbmoon/facenet/blob/master/loss.py
"""
from typing import Dict

import torch
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(torch.nn.Module):

    def __init__(self, margin: float = 1, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)
        self.reduction = reduction

    def forward(self, anchor, positive, negative) -> Dict[str, torch.Tensor]:
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        if self.reduction == 'mean':
            loss = torch.mean(hinge_dist)
        elif self.reduction == 'sum':
            loss = torch.sum(hinge_dist)
        else:
            raise AssertionError('reduction should be mean or sum...')
        return {'loss': loss,
                'neg_dist': neg_dist.detach(),
                'pos_dist': pos_dist.detach()}
