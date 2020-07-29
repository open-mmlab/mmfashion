import torch
import torch.nn as nn

from ..registry import FEATURENORM

@FEATURENORM.register_module
class FeatureL2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super(FeatureL2Norm, self).__init__()
        self.eps = eps

    def forward(self, feature):
        norm_feat = torch.sum(torch.pow(feature, 2), 1) + self.eps
        norm_feat = torch.pow(norm_feat, 0.5).unsqueeze(1)
        norm_feat = norm_feat.expand_as(feature)
        norm_feat = torch.div(feature, norm_feat)
        return norm_feat