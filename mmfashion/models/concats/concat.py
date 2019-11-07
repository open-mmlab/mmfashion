import torch
import torch.nn as nn

from ..registry import CONCATS


@CONCATS.register_module
class Concat(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(Concat, self).__init__()
        # concat global and local
        self.fc_fusion = nn.Linear(inchannels, outchannels)

    def forward(self, global_x, local_x=None):
        if local_x is not None:
            x = torch.cat((global_x, local_x), 1)
            x = self.fc_fusion(x)
        else:
            x = global_x

        return x

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_fusion.weight)
        if self.fc_fusion.bias is not None:
            self.fc_fusion.bias.data.fill_(0.01)
