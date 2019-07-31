import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class TripletLoss(nn.Module):

    def __init__(self, margin, use_sigmoid=True, size_average=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.use_sigmoid = use_sigmoid
        self.size_average = size_average

    def forward(self, anchor, pos, neg):
        if self.use_sigmoid:
            anchor, pos, neg = F.sigmoid(anchor), F.sigmoid(pos), F.sigmoid(
                neg)

        dist_pos = (anchor - pos).pow(2).sum(1)
        dist_neg = (anchor - neg).pow(2).sum(1)
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses
