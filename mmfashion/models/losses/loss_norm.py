import numpy as np
import torch.nn as nn

from ..registry import LOSSES


@LOSSES.register_module
class L2NormLoss(nn.Module):

    def __init__(self, loss_weight=5e-4):
        super(L2NormLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x1, x2, x3, length):
        x_norm = (x1 + x2 + x3) / 3
        loss_norm = x_norm / np.sqrt(length)
        return self.loss_weight * loss_norm


@LOSSES.register_module
class L1NormLoss(nn.Module):

    def __init__(self, loss_weight=5e-4, average=True):
        super(L1NormLoss, self).__init__()
        self.loss_weight = loss_weight
        self.average = average

    def forward(self, x1, x2, x3, length):
        loss_norm = (x1 + x2 + x3) / 3
        if self.average:
            loss_norm = loss_norm / length

        return self.loss_weight * loss_norm
