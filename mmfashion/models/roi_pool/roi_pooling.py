from __future__ import division

import torch
import torch.nn as nn

from ..registry import ROIPOOLING

@ROIPOOLING.register_module
class RoIPooling(nn.Module):
    def __init__(self, pool_plane, inter_plane, outplanes, crop_size=7, img_size=(224,224), num_lms=8, roi_size=2):
        super(RoIPooling, self).__init__()
        self.maxpool = nn.MaxPool2d(pool_plane)
        self.linear = nn.Linear(num_lms*inter_plane, outplanes)
        
        self.inter_plane = inter_plane
        self.outplanes = outplanes
        self.num_lms = num_lms
        self.crop_size = crop_size
        self.img_width = img_size[0]
        self.img_height = img_size[1]
        self.roi_size = roi_size
      
    def _single_ROI(self, x, landmark):
        cropped_x = []
        for i,cor in enumerate(landmark):
            if i%2==0: # x
               if cor==0 and landmark[i+1]==0:
                  x1, y1, x2, y2 = 0,0, self.roi_size, self.roi_size
               else:
                  x1, y1 = max(int(cor/self.img_width*self.crop_size) -1, 0), max(int(landmark[i+1]/self.img_height*self.crop_size)-1, 0)
                  x2, y2 = x1+self.roi_size, y1+self.roi_size
                  if x2>=self.crop_size:
                     x1, x2 = self.crop_size-self.roi_size, self.crop_size
                  if y2>=self.crop_size:
                     y1, y2 = self.crop_size-self.roi_size, self.crop_size
               cropped_x.append(x[:,x1:x2, y1:y2])
            else:
               continue
        cropped_x = torch.cat(cropped_x).view(self.num_lms, -1, self.roi_size, self.roi_size)
      
        return cropped_x

   
    def forward(self, features, landmarks):
        pooled = []
        batch_size = features.size(0)
        for i,x in enumerate(features):
            cropped_x = self._single_ROI(x, landmarks[i])
            single_pool = self.maxpool(cropped_x)
            single_pool = single_pool.view(-1)
            pooled.append(single_pool)
       
        pooled = torch.stack(pooled) 
        pooled = self.linear(pooled)
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
  
