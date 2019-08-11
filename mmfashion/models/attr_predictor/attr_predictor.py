import torch
import torch.nn as nn

from .. import builder
from ..registry import ATTRPREDICTOR

@ATTRPREDICTOR.register_module
class AttrPredictor(nn.Module):
    def __init__(self, 
                 inchannels, 
                 outchannels,
                 loss_attr=dict(
                          type='BCEWithLogitsLoss',
                          weight=None,
                          size_average=None,
                          reduce=None,
                          reduction='mean')):
       super(AttrPredictor, self).__init__()
       self.linear = nn.Linear(inchannels, outchannels)
       self.loss_attr = builder.build_loss(loss_attr)


    def forward(self, x, target=None, train=False):
        x = self.linear(x)
        if train:
           loss_attr = self.loss_attr(x, target)
           return loss_attr
        else:
           return x
    
    def init_weights(self):
        nn.init.normal_(self.linear, 0, 0.01) 
