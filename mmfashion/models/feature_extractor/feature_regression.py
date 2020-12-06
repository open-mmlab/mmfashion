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

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(inter_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels[0], inter_channels[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(inter_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels[1], inter_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels[2], inter_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels[3]),
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Linear(inter_channels[3]*4*3, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x

    def init_weights(self, pretrained):
        if pretrained is not None:
            load_checkpoint(self, pretrained)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif isinstance(m, nn.Linear):
                    nn.init.normal(m.weight.data, 0.0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)