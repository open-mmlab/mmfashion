import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import ATTRPREDICTOR

@ATTRPREDICTOR.register_module
class AttrPredictor(nn.Module):
    def __init__(self, 
                 inchannels, 
                 outchannels):
       super(AttrPredictor, self).__init__()
       self.linear_attr = nn.Linear(inchannels, outchannels)
 
    def forward(self, x):
        attr_pred = self.linear_attr(x)
        return attr_pred

    def init_weights(self):
        nn.init.normal_(self.linear, 0, 0.01) 
