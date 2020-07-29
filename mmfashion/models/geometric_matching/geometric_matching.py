import torch
import torch.nn as nn

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
                 pretrained=None):
        
        super(GeometricMatching, self).__init__()
        
        self.feature_extractora_a = builder.build_feature_extractor(feature_extractor_a)
        self.feature_extractora_b = builder.build_feature_extractor(feature_extractor_b)
        
        self.feature_norm = builder.build_feature_norm(feature_norm)
        self.feature_correlation = builder.build_feature_correlation(feature_correlation)
        self.feature_regression = builder.build_feature_regression(feature_regression)
        
        self.tps_warp = builder.build_tps_warp(tps_warp)
        
        self.init_weights(pretrained=pretrained)

    def forward(self, a, b):
        feat_a = self.feature_extractora_a(a)
        feat_b = self.feature_extractora_b(b)
        feat_a = self.feature_norm(feat_a)
        feat_b = self.feature_norm(feat_b)
        correlation = self.feature_correlation(feat_a, feat_b)

        theta = self.feature_regression(correlation)
        grid = self.tps_warp(theta)
        return grid, theta
    

    def init_weights(self, pretrained=None):
        self.feature_extractora_a.init_weights(pretrained)
        self.feature_extractora_b.init_weights(pretrained)