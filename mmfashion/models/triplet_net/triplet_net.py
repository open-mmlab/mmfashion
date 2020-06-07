from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..builder import build_loss
from ..registry import TRIPLETNET


class EmbedBranch(nn.Module):

    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # L2 normalize each feature vector
        norm = torch.norm(x, p=2, dim=1) + 1e-10
        norm.unsqueeze_(1)
        x = x / norm.expand_as(x)
        return x

    # def init_weights(self):
    #     for m in self.fc1:
    #         if type(m) == nn.Linear:
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 m.bias.data.fill_(0.01)
    #     nn.init.xavier_uniform_(self.fc2.weight)
    #     if self.fc2.bias is not None:
    #         self.fc2.bias.data.fill_(0.01)


@TRIPLETNET.register_module
class TripletNet(nn.Module):

    def __init__(self,
                 text_feature_dim,
                 embed_feature_dim,
                 loss_vse=dict(
                     type='L1NormLoss', loss_weight=5e-3, average=False),
                 loss_triplet=dict(
                     type='MarginRankingLoss', margin=0.3, loss_weight=1),
                 loss_sim_i=dict(
                     type='MarginRankingLoss', margin=0.3, loss_weight=5e-5),
                 loss_selective_margin=dict(
                     type='SelectiveMarginLoss', margin=0.3, loss_weight=5e-5),
                 learned_metric=True):
        super(TripletNet, self).__init__()
        self.text_feature_dim = text_feature_dim
        self.embed_feature_dim = embed_feature_dim
        self.text_branch = EmbedBranch(text_feature_dim, embed_feature_dim)
        self.metric_branch = None

        if learned_metric:
            self.metric_branch = nn.Linear(embed_feature_dim, 1, bias=False)

        self.loss_vse = build_loss(loss_vse)
        self.loss_triplet = build_loss(loss_triplet)
        self.loss_sim_i = build_loss(loss_sim_i)
        self.loss_selective_margin = build_loss(loss_selective_margin)

    def image_forward(self, general_x, general_y, general_z):
        """ calculate image similarity loss on the general embedding
            general_x: general feature extracted by backbone
            general_y: far data(Negative)
            general_z: close data(Positive)
        """
        disti_p = F.pairwise_distance(general_y, general_z, 2)
        disti_n1 = F.pairwise_distance(general_y, general_x, 2)
        disti_n2 = F.pairwise_distance(general_z, general_x, 2)
        target = torch.FloatTensor(disti_p.size()).fill_(1).cuda()
        loss_sim_i1 = self.loss_sim_i(disti_p, disti_n1, target)
        loss_sim_i2 = self.loss_sim_i(disti_p, disti_n2, target)
        loss_sim_i = (loss_sim_i1 + loss_sim_i2) / 2.
        return loss_sim_i

    def embed_forward(self, embed_x, embed_y, embed_z):
        """embed_x, mask_norm_x: type_specific net output (Anchor)
           embed_y, mask_norm_y: type_specifc net output (Negative)
           embed_z, mask_norm_z: type_specifi net output (Positive)
           conditions: only x(anchor data) has conditions
        """
        if self.metric_branch is None:
            dist_neg = F.pairwise_distance(embed_x, embed_y, 2)
            dist_pos = F.pairwise_distance(embed_x, embed_z, 2)
        else:
            dist_neg = self.metric_branch(embed_x * embed_y)
            dist_pos = self.metric_branch(embed_x * embed_z)

        target = torch.FloatTensor(dist_neg.size()).fill_(1)
        target = Variable(target.cuda())

        # type specific triplet loss
        loss_type_triplet = self.loss_triplet(dist_neg, dist_pos, target)
        return loss_type_triplet

    def text_forward(self, text_x, text_y, text_z, has_text_x, has_text_y,
                     has_text_z):
        desc_x = self.text_branch(text_x)
        desc_y = self.text_branch(text_y)
        desc_z = self.text_branch(text_z)

        distd_p = F.pairwise_distance(desc_y, desc_z, 2)
        distd_n1 = F.pairwise_distance(desc_x, desc_y, 2)
        distd_n2 = F.pairwise_distance(desc_x, desc_z, 2)
        has_text = has_text_x * has_text_y * has_text_z
        loss_sim_t1 = self.loss_selective_margin(distd_p, distd_n1, has_text)
        loss_sim_t2 = self.loss_selective_margin(distd_p, distd_n2, has_text)
        loss_sim_t = (loss_sim_t1 + loss_sim_t2) / 2.
        return loss_sim_t, desc_x, desc_y, desc_z

    def calc_vse_loss(self, desc_x, general_x, general_y, general_z, has_text):
        """ Both y and z are assumed to be negatives because they are not from the same
            item as x
            desc_x: Anchor language embedding
            general_x: Anchor visual embedding
            general_y: Visual embedding from another item from input triplet
            general_z: Visual embedding from another item from input triplet
            has_text: Binary indicator of whether x had a text description
        """
        distd1_p = F.pairwise_distance(general_x, desc_x, 2)
        distd1_n1 = F.pairwise_distance(general_y, desc_x, 2)
        distd1_n2 = F.pairwise_distance(general_z, desc_x, 2)
        loss_vse_1 = self.loss_selective_margin(distd1_p, distd1_n1, has_text)
        loss_vse_2 = self.loss_selective_margin(distd1_p, distd1_n2, has_text)
        return (loss_vse_1 + loss_vse_2) / 2.

    def forward(self, general_x, type_embed_x, text_x, has_text_x, general_y,
                type_embed_y, text_y, has_text_y, general_z, type_embed_z,
                text_z, has_text_z):
        """x: Anchor data
           y: Distant(negative) data
           z: Close(positive) data
        """
        loss_sim_i = self.image_forward(general_x, general_y, general_z)
        loss_type_triplet = self.embed_forward(type_embed_x, type_embed_y,
                                               type_embed_z)

        loss_sim_t, desc_x, desc_y, desc_z = self.text_forward(
            text_x, text_y, text_z, has_text_x, has_text_y, has_text_z)

        loss_vse_x = self.calc_vse_loss(desc_x, general_x, general_y,
                                        general_z, has_text_x)
        loss_vse_y = self.calc_vse_loss(desc_y, general_y, general_x,
                                        general_z, has_text_y)
        loss_vse_z = self.calc_vse_loss(desc_z, general_z, general_x,
                                        general_y, has_text_z)
        loss_vse = self.loss_vse(loss_vse_x, loss_vse_y, loss_vse_z,
                                 len(general_x))

        return loss_type_triplet, loss_sim_t, loss_vse, loss_sim_i

    def init_weights(self):
        # self.text_branch.init_weights()
        if self.metric_branch is not None:
            weight = torch.zeros(1, self.embed_feature_dim) / float(
                self.embed_feature_dim)
            self.metric_branch.weight = nn.Parameter(weight)
