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
       self.linear_attr = nn.Linear(inchannels, outchannels[0])
       self.linear_cate = nn.Linear(inchannels, outchannels[1])
 
    def forward(self, x, attr=None, cate=None, train=False):
        attr_pred = self.linear_attr(x)
        cate_pred = self.linear_cate(x)
        return attr_pred, cate_pred
 
    def init_weights(self):
        nn.init.normal_(self.linear, 0, 0.01) 
