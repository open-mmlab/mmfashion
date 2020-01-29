import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import ATTRPREDICTOR


@ATTRPREDICTOR.register_module
class AttrPredictor(nn.Module):

    def __init__(self,
                 inchannels,
                 outchannels,
                 loss_attr=dict(
                     type='BCEWithLogitsLoss',
                     ratio=1,
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean')):
        super(AttrPredictor, self).__init__()
        self.linear_attr = nn.Linear(inchannels, outchannels)
        self.loss_attr = build_loss(loss_attr)

    def forward_train(self, x, attr):
        attr_pred = self.linear_attr(x)
        loss_attr = self.loss_attr(attr_pred, attr)
        return loss_attr

    def forward_test(self, x):
        attr_pred = torch.sigmoid(self.linear_attr(x))
        return attr_pred

    def forward(self, x, attr=None, return_loss=False):
        if return_loss:
            return self.forward_train(x, attr)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_attr.weight)
        if self.linear_attr.bias is not None:
            self.linear_attr.bias.data.fill_(0.01)
