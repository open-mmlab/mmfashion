import torch
import torch.nn as nn
import torch.nn.functional as F

from ..import builder
from ..registry import GEOMETRICMATCHING

@GEOMETRICMATCHING.register_module
class GeometricMatching(nn.Module):
    def __init__(self,
                 feature_extractor_a,
                 feature_extractor_b,
                 feature_norm,
                 feature_correlation,
                 feature_regression,
                 tps_warp,
                 loss=dict(type='L1Loss'),
                 pretrained=None):
        
        super(GeometricMatching, self).__init__()
        
        self.feature_extractora_a = builder.build_feature_extractor(feature_extractor_a)
        self.feature_extractora_b = builder.build_feature_extractor(feature_extractor_b)
        
        self.feature_norm = builder.build_feature_norm(feature_norm)
        self.feature_correlation = builder.build_feature_correlation(feature_correlation)
        self.feature_regression = builder.build_feature_regression(feature_regression)
        
        self.tps_warp = builder.build_tps_warp(tps_warp)

        self.loss = builder.build_loss(loss)
        self.init_weights(pretrained=pretrained)

    def forward_feature(self, a, b):
        feat_a = self.feature_extractora_a(a)
        feat_b = self.feature_extractora_b(b)
        feat_a = self.feature_norm(feat_a)
        feat_b = self.feature_norm(feat_b)
        correlation = self.feature_correlation(feat_a, feat_b)

        theta = self.feature_regression(correlation)
        grid = self.tps_warp(theta)
        return grid, theta
    
    def forward_train(self, agnostic, cloth, parse_cloth):
        grid, theta = self.forward_feature(agnostic, cloth)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')

        # compute loss
        losses = dict()
        losses['gmm_loss'] = self.loss(warped_cloth, parse_cloth)
        return losses

    def forward_test(self, agnostic, cloth, cloth_mask):
        grid, theta = self.forward_feature(agnostic, cloth)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
        warped_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros')

        return warped_cloth, warped_mask

    def forward(self,
                cloth,
                cloth_mask,
                agnostic,
                parse_cloth,
                c_name=None,
                img=None,
                return_loss=True):
        if return_loss:
            return self.forward_train(agnostic, cloth, parse_cloth)
        else:
            return self.forward_test(agnostic, cloth, cloth_mask)

    def init_weights(self, pretrained=None):
        self.feature_extractora_a.init_weights(pretrained)
        self.feature_extractora_b.init_weights(pretrained)
        self.feature_regression.init_weights(pretrained)