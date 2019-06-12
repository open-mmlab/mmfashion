import torch
import torch.nn as nn

from ..registry import CONCATS

@CONCATS.register_module
class Concat(nn.Module):
   
    def __init__(self, num_classes):
        super(Concat, self).__init__()
        self.fc_fusion = nn.Linear(2*4096, 4096)
        self.fc = nn.Linear(4096, num_classes)
    
    def forward(self, global_x, local_x=None):
        if local_x is not None:
           x = torch.cat((global_x, local_x), 1)
           x = self.fc_fusion(x)
        else:
           x = global_x
        x = self.fc(x)
        return x
