import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from models.config import cfg
from models.layer_utils.ROIPool import ROIPooling
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=False, reduce=False):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()  # batch_size * 88

    if weight is not None:
        loss = loss * weight
    
    if not reduce:
        return loss.mean()
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

    
class WeightedBCELoss(nn.Module):
    def __init__(self, weight=None, PosWeightIsDynamic= True, WeightIsDynamic= False, size_average=False, reduce=False):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super(WeightedBCELoss, self).__init__()

        self.register_buffer('weight', weight)

        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

        
    def forward(self, input, target):

        positive_counts = target.sum(dim=0)
        nBatch = len(target)
        pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)
            
        if self.weight is not None:
            return weighted_binary_cross_entropy(input, target,
                                                 pos_weight,
                                                 weight=self.weight,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy(input, target,
                                                 pos_weight,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)


