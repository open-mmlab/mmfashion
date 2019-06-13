import logging

import torch
import torch.nn as nn

from .base import BasePredictor
from .. import builder
from ..registry import PREDICTORS

@PREDICTORS.register_module
class RoIPredictor(BasePredictor):

    def __init__(self, 
                 backbone, 
                 global_pool, 
                 concat,
                 roi_pool=None, 
                 pretrained=None):
        super(BasePredictor, self).__init__()
  
        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
       
        if roi_pool is not None:
           self.roi_pool = builder.build_roi_pool(roi_pool)
        
        self.concat = builder.build_concat(concat)
        

    def forward_train(self, x, landmarks=None, iuv=None):
        # 1. conv layers extract features
        x = self.backbone(x)
        
        # 2. global pooling
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
  
        # 3. roi pooling
        if self.with_roi_pool:
           local_x = self.roi_pool(x, landmarks)        
        else:
           local_x = None

        # 4. concat
        pred = self.concat(global_x, local_x)
        return pred

    def simple_test(self, x, landmarks=None, iuv=None):
        """Test single image"""
        x = x.unsqueeze(0)
        if landmarks is not None:
           landmarks = landmarks.unsqueeze(0)
        if iuv is not None:
           iuv = iuv.unsqueeze(0)

        pred = self.aug_test(x, landmarks, iuv)[0]
        return pred

    def aug_test(self, x, landmarks=None, iuv=None):
        """Test batch of images"""
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        if self.with_roi_pool:
           local_x = self.roi_pool(x, landmarks)
        else:
           local_x = None
        
        pred = self.concat(global_x, local_x)
        return pred


    def init_weights(self, pretrained=None):
        print('base predictors load weights')
        super(RoIPredictor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights(pretrained=pretrained)

        if self.with_roi_pool:
           self.roi_pool.init_weights()
  
        self.concat.init_weights(pretrained=pretrained)
