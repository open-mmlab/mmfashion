import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import CATEPREDICTOR


@CATEPREDICTOR.register_module
class CatePredictor(nn.Module):

    def __init__(self,
                 inchannels,
                 outchannels,
                 loss_cate=dict(
                     type='CELoss',
                     ratio=1,
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean')):
        super(CatePredictor, self).__init__()
        self.linear_cate = nn.Linear(inchannels, outchannels)
        self.loss_cate = build_loss(loss_cate)

    def forward_train(self, x, cate):
        cate_pred = self.linear_cate(x)
        loss_cate = self.loss_cate(cate_pred, cate)
        return loss_cate

    def forward_test(self, x):
        cate_pred = torch.sigmoid(self.linear_cate(x))
        return cate_pred

    def forward(self, x, cate=None, return_loss=False):
        if return_loss:
            return self.forward_train(x, cate)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_cate.weight)
        if self.linear_cate.bias is not None:
            self.linear_cate.bias.data.fill_(0.01)
