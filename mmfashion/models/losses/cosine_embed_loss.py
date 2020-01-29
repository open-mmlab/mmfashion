import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class CosineEmbeddingLoss(nn.Module):

    def __init__(self,
                 margin=0.,
                 size_average=None,
                 reduce=None,
                 reduction='mean'):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(
            input1,
            input2,
            target,
            margin=self.margin,
            reduction=self.reduction)
