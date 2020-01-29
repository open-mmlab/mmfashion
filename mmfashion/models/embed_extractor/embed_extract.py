import torch.nn as nn

from ..builder import build_loss
from ..registry import EMBEDEXTRACTOR


@EMBEDEXTRACTOR.register_module
class EmbedExtractor(nn.Module):

    def __init__(self,
                 inchannels,
                 inter_channels,
                 loss_id=dict(
                     type='CELoss',
                     ratio=1,
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean'),
                 loss_triplet=dict(type='TripletLoss', method='cosine')):
        super(EmbedExtractor, self).__init__()
        self.embed_linear = nn.Linear(inchannels, inter_channels[0])
        self.bn = nn.BatchNorm1d(inter_channels[0], inter_channels[1])
        self.id_linear = nn.Linear(inter_channels[0], inter_channels[1])

        self.loss_id = build_loss(loss_id)
        if loss_triplet is not None:
            self.loss_triplet = build_loss(loss_triplet)
        else:
            self.loss_triplet = None

    def forward_train(self, x, id, triplet, pos, neg, triplet_pos_label,
                      triplet_neg_label):

        embed = self.embed_linear(x)
        id_pred = self.id_linear(embed)

        loss_id = self.loss_id(id_pred, id)
        if triplet:
            pos_embed = self.embed_linear(pos)
            neg_embed = self.embed_linear(neg)
            loss_triplet = self.loss_triplet(embed, pos_embed, neg_embed,
                                             triplet_pos_label,
                                             triplet_neg_label)
            return loss_id + loss_triplet
        else:
            return loss_id

    def forward_test(self, x):
        embed = self.embed_linear(x)
        return embed

    def forward(self,
                x,
                id,
                return_loss=False,
                triplet=False,
                pos=None,
                neg=None,
                triplet_pos_label=None,
                triplet_neg_label=None):
        if return_loss:
            return self.forward_train(x, id, triplet, pos, neg,
                                      triplet_pos_label, triplet_neg_label)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed_linear.weight)
        if self.embed_linear.bias is not None:
            self.embed_linear.bias.data.fill_(0.01)

        nn.init.xavier_uniform_(self.id_linear.weight)
        if self.id_linear.bias is not None:
            self.id_linear.bias.data.fill_(0.01)
