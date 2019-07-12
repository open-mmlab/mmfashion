from __future__ import division

import torch
import torch.nn as nn

from ..registry import GLOBALPOOLING

@GLOBALPOOLING.register_module
class GlobalPooling(nn.Module):
    def __init__(self, inplanes, pool_plane, inter_plane, outplanes):
        super(GlobalPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(inplanes)
        self.maxpool = nn.MaxPool2d(pool_plane)
        self.classifier =  nn.Sequential(
            nn.Linear(inter_plane, outplanes),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(outplanes, outplanes),
            nn.ReLU(True),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        global_pool = self.classifier(x)
        return global_pool
        
