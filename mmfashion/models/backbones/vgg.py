import torch.nn as nn
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES


@BACKBONES.register_module
class Vgg(nn.Module):
    setting = {
        'vgg16': [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512, 'M'
        ],
    }

    def __init__(self,
                 layer_setting='vgg16',
                 batch_norm=False,
                 init_weights=False):
        super(Vgg, self).__init__()
        self.features = self._make_layers(self.setting[layer_setting],
                                          batch_norm)

        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x

    def init_weights(self, pretrained=None):
        print('pretrained model', pretrained)
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
