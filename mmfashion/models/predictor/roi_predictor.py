import logging

import torch
import torch.nn as nn

from .base import BasePredictor
from .. import builder
from ..registry import PREDICTOR


@PREDICTOR.register_module
class RoIPredictor(BasePredictor):

    def __init__(self,
                 backbone,
                 global_pool,
                 concat,
                 loss_cate,
                 loss_attr,
                 roi_pool=None,
                 pretrained=None):
        super(BasePredictor, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)

        assert roi_pool is not None
        self.roi_pool = builder.build_roi_pool(roi_pool)

        self.concat = builder.build_concat(concat)
        self.loss_attr = builder.build_loss(loss_attr)
        self.loss_cate = builder.build_loss(loss_cate)


    def forward_train(self, x, attr, cate, landmarks):
        # 1. conv layers extract global features
        x = self.backbone(x)

        # 2. global pooling
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)

        # 3. roi pooling
        local_x = self.roi_pool(x, landmarks)

        # 4. concat
        attr_pred, cate_pred = self.concat(global_x, local_x)
        losses = dict()
        cate = cate.view(-1)
        losses['loss_cate'] = self.loss_cate(cate_pred, cate)
        losses['loss_attr'] = self.loss_attr(attr_pred, attr)

        return losses


    def simple_test(self, x, landmarks=None):
        """Test single image"""
        x = x.unsqueeze(0)
        landmarks = landmarks.unsqueeze(0)

        attr_pred, cate_pred = self.aug_test(x, landmarks)[0]
        return pred


    def aug_test(self, x, landmarks=None):
        """Test batch of images"""
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        local_x = self.roi_pool(x, landmarks)

        attr_pred, cate_pred = self.concat(global_x, local_x)
        return attr_pred, cate_pred

   
    def forward_test(self, x, landmarks):
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        local_x = self.roi_pool(x, landmarks)

        attr_pred, cate_pred = self.concat(global_x, local_x)
        return attr_pred, cate_pred


    def init_weights(self, pretrained=None):
        super(RoIPredictor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights(pretrained=pretrained)
        self.roi_pool.init_weights()
        self.concat.init_weights(pretrained=pretrained)
