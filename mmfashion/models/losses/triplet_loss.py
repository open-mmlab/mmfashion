import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class TripletLoss(nn.Module):

    def __init__(self, 
                 method='cosine',
                 ratio=5,
                 margin=0.2, 
                 use_sigmoid=False, 
                 size_average=True):
        super(TripletLoss, self).__init__()
        self.method = method
        self.ratio = ratio
        self.margin = margin
        self.use_sigmoid = use_sigmoid
        self.size_average = size_average

    def forward(self, anchor, pos, neg):
        if self.use_sigmoid:
            anchor, pos, neg = F.sigmoid(anchor), F.sigmoid(pos), F.sigmoid(
                neg)
        if self.method == 'cosine':
            anchor = torch.cos(anchor)
            pos = torch.cos(pos)
            neg = torch.cos(neg)
            dist_pos = abs(anchor-pos)
            dist_neg = abs(anchor-neg)
            losses = self.ratio * F.relu(dist_pos - dist_neg + self.margin)
        else:
           dist_pos = (anchor - pos).pow(2).sum(1)
           dist_neg = (anchor - neg).pow(2).sum(1)
           losses = self.ratio * F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean() if self.size_average else losses.sum()
