import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ..registry import FEATUREEXTRACTOR

@FEATUREEXTRACTOR.register_module
class FeatureExtractor(nn.Module):

    def __init__(self,
                 in_channels,
                 ngf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d):
        super(FeatureExtractor, self).__init__()
        # downsample layer
        downconv = nn.Conv2d(in_channels, ngf, kernel_size=4, stride=2, padding=1)
        layers = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            if 2**i * ngf < 512:
                in_ngf = 2**i * ngf
                out_ngf = 2**(i+1) * ngf
            else:
                in_ngf = 512
                out_ngf = 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            layers += [downconv, nn.ReLU(True), norm_layer(out_ngf)]
        # last layer in feature extractor
        downconv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        layers += [downconv, nn.ReLU(True), norm_layer(512)]
        layers += [downconv, nn.ReLU(True)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def init_weights(self, pretrained=None):
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
