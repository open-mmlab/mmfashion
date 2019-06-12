import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES

def cross_entropy(pred, label, weight, reduction=True):
    return F.cross_entropy(pred, label, weight=weight, reduction=None)


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):
    def __init__(self, use_sigmoid=False, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.weight = weight
        self.reduction = self.reduction
        self.cls_criterion = cross_entropy

        
    def forward(self, pred, label, *args, **kwargs):
        if self.use_sigmoid:
           pred = F.sigmoid(pred)
        return self.cls_criterion(pred, label, self.weight, *args, **kwargs)
