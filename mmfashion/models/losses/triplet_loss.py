import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class TripletLoss(nn.Module):

    def __init__(self,
                 method='cosine',
                 ratio=1,
                 margin=0.2,
                 use_sigmoid=False,
                 reduction='mean',
                 size_average=True):
        super(TripletLoss, self).__init__()
        self.method = method
        self.ratio = ratio
        self.margin = margin
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.size_average = size_average

    def forward(self, anchor, pos, neg, triplet_pos_label, triplet_neg_label):
        if self.use_sigmoid:
            anchor, pos, neg = F.sigmoid(anchor), F.sigmoid(pos), F.sigmoid(
                neg)
        if self.method == 'cosine':  # cosine similarity loss
            loss_pos = F.cosine_embedding_loss(
                anchor,
                pos,
                triplet_pos_label,
                margin=self.margin,
                reduction=self.reduction)

            loss_neg = F.cosine_embedding_loss(
                anchor,
                neg,
                triplet_neg_label,
                margin=self.margin,
                reduction=self.reduction)
            losses = loss_pos + loss_neg

        else:  # L2 loss
            dist_pos = (anchor - pos).pow(2).sum(1)
            dist_neg = (anchor - neg).pow(2).sum(1)
            losses = self.ratio * F.relu(dist_pos - dist_neg + self.margin)
            if self.size_average:
                losses = losses.mean()
            else:
                losses = losses.sum()
        return losses
