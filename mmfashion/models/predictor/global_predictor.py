import logging

import torch
import torch.nn as nn

from .base import BasePredictor
from .. import builder
from ..registry import PREDICTOR


@PREDICTOR.register_module
class GlobalPredictor(BasePredictor):

    def __init__(self,
                 backbone,
                 global_pool,
                 concat,
                 loss=dict(
                     type='BCEWithLogitsLoss',
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean'),
                 roi_pool=None,
                 pretrained=None):
        super(BasePredictor, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
      
        assert roi_pool is None

        self.concat = builder.build_concat(concat)
        self.loss = builder.build_loss(loss)


    def forward_train(self, x, label, landmarks):
        assert landmarks is None
        # 1. conv layers extract global features
        x = self.backbone(x)

        # 2. global pooling
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)

        # 4. concat
        pred = self.concat(global_x)

        losses = dict()
        loss = self.loss(pred, label)
        losses['loss'] = loss

        return losses


    def simple_test(self, x, landmarks):
        """Test single image"""
        assert landmarks is None
        x = x.unsqueeze(0)
        pred = self.aug_test(x)[0]
        return pred


    def aug_test(self, x, landmarks):
        """Test batch of images"""
        assert landmarks is None
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        pred = self.concat(global_x)
        return pred


    def forward_test(self, x, landmarks):
        assert landmarks is None
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        pred = self.concat(global_x)
        return pred


    def init_weights(self, pretrained=None):
        super(RoIPredictor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights(pretrained=pretrained)
        self.concat.init_weights(pretrained=pretrained)
