import torch
import torch.nn as nn

from ..registry import CONCATS

@CONCATS.register_module
class Concat(nn.Module):
   
    def __init__(self, inplanes, inter_plane, num_classes, retrieve=False):
        super(Concat, self).__init__()
        self.fc_fusion = nn.Linear(inplanes, inter_plane)
        self.fc = nn.Linear(inter_plane, num_classes)
        self.retrieve = retrieve

    def forward(self, global_x, local_x=None):
        if local_x is not None:
           x = torch.cat((global_x, local_x), 1)
           x = self.fc_fusion(x)
        else:
           x = global_x
        
        if self.retrieve:
           return x
        
        x = self.fc(x)
        return x
