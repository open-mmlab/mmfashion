import torch
import torch.nn as nn

from ..registry import FEATUREREGRESSION

@FEATUREREGRESSION.register_module
class FeatureRegression(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=6,
                 inter_channels=(512, 256, 128, 64)):
        super(FeatureRegression, self).__init__()

        conv_layers = []
        conv_layer = nn.Conv2d(in_channels, inter_channels[0],
                               kernel_size=4, stride=2, padding=1)
        conv_layers += [conv_layer, nn.BatchNorm2d(inter_channels[0]), nn.ReLU(True)]

        if len(inter_channels) > 2:
            for i, inter_channel in enumerate(inter_channels[1:-1]):
                conv_layer = nn.Conv2d(inter_channel, inter_channels[i+1],
                                       kernel_size=4, stride=2, padding=1)
                conv_layers += [conv_layer, nn.BatchNorm2d(inter_channels[0]), nn.ReLU(True)]
        elif len(inter_channels) == 2:
            conv_layer = nn.Conv2d(inter_channels[0], inter_channels[1],
                                   kernel_size=4, stride=2, padding=1)
            conv_layers += [conv_layer, nn.BatchNorm2d(inter_channels[0]), nn.ReLU(True)]

        self.conv_layers = nn.Sequential(*conv_layers)

        self.linear = nn.Linear(inter_channels[-1]*4*3, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x