import torch
import torch.nn as nn

from ..registry import CONCATS


@CONCATS.register_module
class Concat(nn.Module):

    def __init__(self, inchannels, inter_channels, num_attr, num_cate=48, retrieve=False):
        super(Concat, self).__init__()
        # concat global and local
        self.fc_fusion = nn.Linear(inchannels, inter_channels[0]) 
        
        # attribute prediction
        self.fc_attr = nn.Linear(inter_channels[0], num_attr)
        
        # project feature embeds to another plane: self.fc_cate[0]
        # category/id prediction: self.fc_cate[1] 
        self.fc_cate = nn.Sequential(
                          nn.Linear(inter_channels[0], inter_channels[1]),
                          nn.Linear(inter_channels[1], num_cate)
                          )

        self.retrieve = retrieve


    def forward(self, global_x, local_x=None):
        if local_x is not None:
            x = torch.cat((global_x, local_x), 1)
            x = self.fc_fusion(x)
        else:
            x = global_x

        attr_pred = self.fc_attr(x)

        if self.retrieve:
            embed = self.fc_cate[0](x)
            id_pred = self.fc_cate[1](embed)
            return embed, attr_pred, id_pred
        else:
            cate_pred = self.fc_cate(x)
            return attr_pred, cate_pred
