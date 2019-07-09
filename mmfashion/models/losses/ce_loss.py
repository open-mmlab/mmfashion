import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES

@LOSSES.register_module
class CrossEntropyLoss(nn.Module):
    def __init__(self, 
                 weight=None, 
                 size_average=None, 
                 ignore_index=-100,
                 reduce=None, 
                 reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

 
    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
