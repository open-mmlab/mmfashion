"""losses
"""
import torch.nn as nn


def weighted_binary_cross_entropy(sigmoid_x,
                                  targets,
                                  pos_weight,
                                  weight=None,
                                  size_average=False,
                                  reduction=False):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class.
            Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample

    Returns:
        loss
    """
    if not targets.size() == sigmoid_x.size():
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                targets.size(), sigmoid_x.size()))
    loss = -pos_weight * targets * sigmoid_x.log() - (1 - targets) * (
        1 - sigmoid_x).log()
    if weight is not None:
        loss = loss * weight
    if not reduction or size_average:
        return loss.mean()
    return loss.sum()


class WeightedBCELoss(nn.Module):
    """WeightedBCELoss
    """

    def __init__(self,
                 weight=None,
                 pos_weight_is_dynamic=True,
                 size_average=False,
                 reduction=False):
        """
        Args:
            weight: Weight for Each class. Size [1,C]
            pos_weight_is_dynamic: If True, the pos_weight is computed on each
                batch. If pos_weight is None, then it remains None.
                pos_weight: Weight for postive samples. Size [1,C]
        """
        super(WeightedBCELoss, self).__init__()
        self.register_buffer('weight', weight)
        self.size_average = size_average
        self.reduction = reduction
        self.pos_weight_is_dynamic = pos_weight_is_dynamic

    def forward(self, input, target):
        """forward

        Returns:
            loss
        """
        positive_counts = target.sum(dim=0)
        n_batch = len(target)
        pos_weight = (n_batch - positive_counts) / (positive_counts + 1e-5)
        if self.weight is not None:
            return weighted_binary_cross_entropy(
                input,
                target,
                pos_weight,
                weight=self.weight,
                size_average=self.size_average,
                reduction=self.reduction)
        return weighted_binary_cross_entropy(
            input,
            target,
            pos_weight,
            weight=None,
            size_average=self.size_average,
            reduction=self.reduction)
