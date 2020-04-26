import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class MarginRankingLoss(nn.Module):

    def __init__(self,
                 margin=0.2,
                 loss_weight=5e-5,
                 size_average=None,
                 reduce=None,
                 reduction='mean'):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, input1, input2, target):
        return self.loss_weight * F.margin_ranking_loss(
            input1,
            input2,
            target,
            margin=self.margin,
            reduction=self.reduction)


@LOSSES.register_module
class SelectiveMarginLoss(nn.Module):

    def __init__(self, loss_weight=5e-5, margin=0.2):
        super(SelectiveMarginLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight

    def forward(self, pos_samples, neg_samples, has_sample):
        margin_diff = torch.clamp(
            (pos_samples - neg_samples) + self.margin, min=0, max=1e6)
        num_sample = max(torch.sum(has_sample), 1)
        return self.loss_weight * (
            torch.sum(margin_diff * has_sample) / num_sample)
