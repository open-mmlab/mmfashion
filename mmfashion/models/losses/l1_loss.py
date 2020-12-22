import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class L1Loss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(L1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)
