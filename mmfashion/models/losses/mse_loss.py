import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class MSELoss(nn.Module):

    def __init__(self,
                 ratio=1,
                 size_average=None,
                 reduce=None,
                 reduction='mean'):
        super(MSELoss, self).__init__()
        self.ratio = ratio
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target, avg_factor=None):
        return self.ratio * F.mse_loss(input, target, reduction=self.reduction)
