import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class BCEWithLogitsLoss(nn.Module):

    def __init__(self, ratio, weight, size_average, reduce, reduction):
        super(BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction

        self.ratio = ratio

    def forward(self, input, target):
        target = target.float()
        return self.ratio * F.binary_cross_entropy_with_logits(
            input, target, self.weight, reduction=self.reduction)
