import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES

def binary_cross_entropy(pred, label, weight, reduction):
    return F.binary_cross_entropy(pred, label, weight=weight, reduction=reduction)

@LOSSES.register_module
class BinaryCrossEntropyLoss(nn.Module):
   
    def __init__(self, use_sigmoid=True, weight=None, reduction='mean'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.weight = weight
        self.reduction = self.reduction
        self.cls_criterion = binary_cross_entropy

    def forward(self, pred, label):
        if self.use_sigmoid:
           pred = F.sigmoid(pred)
        return self.cls_criterion(pred, label, self.weight, self.reduction)
        
