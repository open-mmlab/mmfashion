import torch
import torch.nn as nn

from ..registry import UNETSKIPCONNECTIONBLOCK


@UNETSKIPCONNECTIONBLOCK.register_module
class UnetSkipConnectionBlock(nn.Module):

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(
            input_nc,
            inner_nc,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv, upnorm]
            unet = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(
                inner_nc,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            unet = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                unet = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                unet = down + [submodule] + up

        self.unet = nn.Sequential(*unet)

    def forward(self, x):
        if self.outermost:
            return self.unet(x)
        else:
            return torch.cat([x, self.unet(x)], 1)
