import torch.nn as nn

from ..builder import build_loss
from ..registry import LANDMARKREGRESSION


@LANDMARKREGRESSION.register_module
class LandmarkRegression(nn.Module):

    def __init__(self,
                 inchannels,
                 outchannels,
                 landmark_num,
                 loss_regress=dict(
                     type='MSELoss', ratio=0.0001, reduction='mean')):
        super(LandmarkRegression, self).__init__()
        self.linear = nn.Linear(inchannels, outchannels)
        self.landmark_num = landmark_num
        self.loss_regress = build_loss(loss_regress)

    def forward_train(self, x, pred_vis, vis, landmark):
        pred_lm = self.linear(x).view(-1, self.landmark_num, 2)
        pred_vis = pred_vis.view(-1, self.landmark_num, 1)
        landmark = landmark.view(-1, self.landmark_num, 2)
        vis = vis.view(-1, self.landmark_num, 1)
        loss_regress = self.loss_regress(vis * pred_lm, vis * landmark)
        return loss_regress

    def forward_test(self, x):
        pred_lm = self.linear(x)
        return pred_lm

    def forward(self,
                x,
                pred_vis=None,
                vis=None,
                landmark=None,
                return_loss=True):
        if return_loss:
            return self.forward_train(x, pred_vis, vis, landmark)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.01)
