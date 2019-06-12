from __future__ import division

import torch
import torch.nn as nn

from ..registry import ROIPOOLING

@ROIPOOLING.register_module
class RoIPooling(nn.Module):
    def __init__(self):
        super(RoIPooling, self).__init__()
        self.maxpool = nn.MaxPool2d((2,2))
    
    def _single_ROI(self, x, landmark):
        cropped_x = []
        for i,cor in enumerate(landmark):
            if i%2==0: # x
               if cor==0 and landmark[i+1]==0:
                  x1, y1, x2, y2 = 0,0,2,2
               else:
                  x1, y1 = max(int(cor/224.*7) -1, 0), max(int(landmark[i+1]/224.*7)-1, 0)
                  x2, y2 = x1+2, y1+2
                  if x2>=7:
                     x1, x2 = 5,7
                  if y2>=7:
                     y1, y2 = 5,7
               cropped_x.append(x[:,x1:x2, y1:y2])
            else:
               continue
        cropped_x = torch.cat(cropped_x).view(8, -1, 2, 2)
        return cropped_x

   
    def forward(self, features, landmarks):
        pooled = []
        batch_size = features.size(0)
        for i,x in enumerate(features):
            cropped_x = self._single_ROI(x, landmarks[i])
            single_pool = self.maxpool(cropped_x)
            single_pool = single_pool.view(-1)
            pooled.append(single_pool)
        pooled = torch.cat(pooled).view(batch_size, -1)
        return pooled

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
  
