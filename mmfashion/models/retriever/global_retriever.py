import logging

import torch
import torch.nn as nn

from .base import BaseRetriever
from .. import builder
from ..registry import RETRIEVER


@RETRIEVER.register_module
class GlobalRetriever(BaseRetriever):

    def __init__(self,
                 backbone,
                 global_pool,
                 concat,
                 loss_attr=dict(
                     type='BCEWithLogitsLoss',
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean'),
                 loss_id=dict(
                     type='CELoss',
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean'),
                 loss_retrieve=dict(
                     type='TripletLoss',
                     margin=1.0,
                     use_sigmoid=True,
                     size_average=True),
                 roi_pool=None,
                 pretrained=None):
        super(GlobalRetriever, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
  
        assert roi_pool is None

        self.concat = builder.build_concat(concat)
        self.loss_attr = builder.build_loss(loss_attr)
        self.loss_id = builder.build_loss(loss_id)
        self.loss_retrieve = builder.build_loss(loss_retrieve)


    def extract_feat(self, x):
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        embed, attr_pred, id_pred = self.concat(global_x)
        return embed, attr_pred, id_pred

    def forward_train(self,
                      anchor,
                      id,
                      label,
                      pos,
                      neg,
                      anchor_lm=None,
                      pos_lm=None,
                      neg_lm=None):
        # extract features
        anchor_embed, attr_pred, id_pred = self.extract_feat(anchor)
        pos_embed, _ ,_ = self.extract_feat(pos)
        neg_embed, _ ,_ = self.extract_feat(neg)

        losses = dict()

        losses['loss_attr'] = self.loss_attr(attr_pred, label)
        losses['loss_id'] = self.loss_id(id_pred, id.view(-1))
        losses['loss_retrieve'] = self.loss_retrieve(anchor_embed, pos_embed,
                                                     neg_embed)
        return losses


    def simple_test(self, x, landmarks=None):
        """Test single image"""
        x = x.unsqueeze(0)
        embed, _ = self.extract_feat(x)
        return embed

    def aug_test(self, x, landmarks=None):
        """Test batch of images"""
        embed, _ = self.extract_feat(x)
        return embed

    def init_weights(self, pretrained=None):
        super(RoIRetriever, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights(pretrained=pretrained)
        self.concat.init_weights(pretrained=pretrained)
