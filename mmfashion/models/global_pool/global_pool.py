from __future__ import division

import torch
import torch.nn as nn

from ..registry import GLOBALPOOLING


@GLOBALPOOLING.register_module
class GlobalPooling(nn.Module):

    def __init__(self, inplanes, pool_plane, inter_channels, outchannels ):
        super(GlobalPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(inplanes)

        inter_plane = inter_channels[0]*inplanes[0]*inplanes[1]
       
        self.classifier = nn.Sequential(
            nn.Linear(inter_plane, inter_channels[1]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(inter_channels[1], outchannels),
            nn.ReLU(True),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        global_pool = self.classifier(x)
        return global_pool
