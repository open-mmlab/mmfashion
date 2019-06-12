from __future__ import division

import torch
import torch.nn as nn

from ..registry import GLOBALPOOLING

@GLOBALPOOLING.register_module
class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.maxpool = nn.MaxPool2d((2,2))
        self.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        global_pool = self.classifier(x)
        return global_pool
        
