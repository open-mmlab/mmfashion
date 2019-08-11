import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..registry import EMBEDEXTRACTOR

@EMBEDEXTRACTOR.register_module
class EmbedExtractor(nn.Module):
   def __init__(self,
                inchannels,
                inter_channels,
                loss_id=dict(
                        type='CELoss',
                        weight=None,
                        size_average=None,
                        reduce=None,
                        reduction='mean'),
                loss_triplet=dict(
                        type='TripletLoss',
                        method='cosine')):
       super(EmbedExtractor, self).__init__()
       self.embed_linear = nn.Linear(inchannels, inter_channels[0])
       self.bn = nn.BatchNorm1d(inter_channels[0], inter_channels[1])
       self.id_linear = nn.Linear(inter_channels[0], inter_channels[1])
 
       self.loss_id = builder.build_loss(loss_id)
       if loss_triplet is not None:
          self.loss_triplet = builder.build_loss(loss_triplet)


   def forward_train(self, x, id, triplet, pos, neg):
       losses = dict()
       embed = self.embed_linear(x)
       id_pred = self.id_linear(embed)

       losses['loss_id'] = self.loss_id(id_pred, id)
       if triplet:
          pos_embed = self.embed_linear(pos)
          neg_embed = self.embed_linear(neg)
          losses['loss_triplet'] = self.loss_triplet(embed, pos_embed, neg_embed)
       return losses

   def forward_test(self, x):
       embed = self.embed_linear(x)
       id_pred = self.id_linear(embed)
       return id_pred


   def forward(self, x, id, train=False, triplet=False, pos=None, neg=None):
       if train:
          return forward_train(x, id, triplet, pos, neg)
       else:
          return forward_test(x)          

   def init_weights(self):
       nn.init.xavier_uniform_(self.embed_linear.weight)
       if self.embed_linear.bias is not None:
          self.embed_linear.bias.data.fill_(0.01)

       nn.init.xavier_uniform_(self.id_linear.weight)
       if self.id_linear.bias is not None:
          self.id_linear.bias.data.fill_(0.01)

