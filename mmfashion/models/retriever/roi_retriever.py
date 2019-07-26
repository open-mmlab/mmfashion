import logging

import torch
import torch.nn as nn

from .base import BaseRetriever
from .. import builder
from ..registry import RETRIEVER

@RETRIEVER.register_module
class RoIRetriever(BaseRetriever):

    def __init__(self, 
                 backbone, 
                 global_pool, 
                 concat,
                 loss_cls=dict(
                      type='BCEWithLogitsLoss',
                      weight=None,
                      size_average=None,
                      reduce=None,
                      reduction='mean'),
                 loss_retrieve=dict(
                      type='TripletLoss',
                      margin=1.0,
                      use_sigmoid=True,
                      size_average=True
                      ),
                 roi_pool=None, 
                 pretrained=None):
        super(BaseRetriever, self).__init__()
  
        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
       
        if roi_pool is not None:
           self.roi_pool = builder.build_roi_pool(roi_pool)
        
        self.concat = builder.build_concat(concat)
        self.loss_cls = builder.build_loss(loss_cls)
        self.loss_retrieve = builder.build_loss(loss_retrieve)


    def extract_feat(self, x, landmarks):
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
  
        if landmarks is not None:
           local_x = self.roi_pool(x, landmarks)
        else:
           local_x = None

        embed, pred = self.concat(global_x, local_x)
        return embed, pred


    def forward_train(self, 
                      anchor,
                      label,
                      pos,
                      neg,
                      anchor_lm=None,
                      pos_lm=None, 
                      neg_lm=None):
        # extract features
        anchor_embed, pred = self.extract_feat(anchor, anchor_lm)
        pos_embed, _ = self.extract_feat(pos, pos_lm)
        neg_embed, _ = self.extract_feat(neg, neg_lm) 
        
        losses = dict()
         
        losses['loss_cls'] = self.loss_cls(pred, label)
        losses['loss_retrieve'] = self.loss_retrieve(anchor_embed, pos_embed, neg_embed)
        return losses


    def simple_test(self, x, landmarks=None):
        """Test single image"""
        x = x.unsqueeze(0)
        if landmarks is not None:
           landmarks = landmarks.unsqueeze(0)
        embed, pred = self.extract_feat(x, landmarks)
        return embed


    def aug_test(self, x, landmarks=None):
        """Test batch of images"""
        embed, pred = self.extract_feat(x, landmarks)
        return embed


    def init_weights(self, pretrained=None):
        super(RoIRetriever, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights(pretrained=pretrained)

        if self.with_roi_pool:
           self.roi_pool.init_weights()
  
        self.concat.init_weights(pretrained=pretrained)
