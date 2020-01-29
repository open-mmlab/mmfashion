import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..registry import VISIBILITYCLASSIFIER


@VISIBILITYCLASSIFIER.register_module
class VisibilityClassifier(nn.Module):

    def __init__(self,
                 inchannels,
                 outchannels,
                 landmark_num,
                 loss_vis=dict(
                     type='BCEWithLogitsLoss', ratio=1, reduction='none')):
        super(VisibilityClassifier, self).__init__()
        self.linear = nn.Linear(inchannels, 1)
        self.landmark_num = landmark_num

        self.loss_vis = builder.build_loss(loss_vis)

    def forward_train(self, x, vis):
        losses_vis = []
        vis_pred_list = []
        for i in range(self.landmark_num):
            lm_feat = x[:, i, :]  # landmark feature (bs, 256)
            vis_pred = F.sigmoid(
                self.linear(lm_feat))  # landmark visibility (bs, 2)
            lm_vis = vis[:, i].unsqueeze(1)
            vis_pred_list.append(lm_vis)

            loss_vis = self.loss_vis(vis_pred, lm_vis)
            losses_vis.append(loss_vis)

        losses_vis_tensor = torch.stack(losses_vis).transpose(1, 0)[:, :, 0]
        vis_pred_list = torch.stack(vis_pred_list).transpose(1, 0)[:, :, 0]

        # calculate mean value
        losses_vis_tensor_mean_per_lm = torch.mean(
            losses_vis_tensor, dim=1, keepdim=True)
        losses_vis_tensor_mean_per_batch = torch.mean(
            losses_vis_tensor_mean_per_lm)

        return losses_vis_tensor_mean_per_batch, vis_pred_list

    def forward_test(self, x):
        vis_pred_list = []
        for i in range(self.landmark_num):
            lm_feat = x[:, i, :]
            vis_pred = F.sigmoid(
                self.linear(lm_feat))  # landmark visibility (bs, 2)
            vis_pred_list.append(vis_pred)
        vis_pred_list = torch.stack(vis_pred_list).transpose(1, 0)[:, :, 0]
        return vis_pred_list

    def forward(self, x, vis=None, return_loss=True):
        if return_loss:
            return self.forward_train(x, vis)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.01)
