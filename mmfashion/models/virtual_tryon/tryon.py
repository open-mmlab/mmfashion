import torch
import torch.nn as nn
import torch.nn.functional as F

from ..import builder
from ..registry import TRYON

@TRYON.register_module
class Tryon(nn.Module):
    def __init__(self,
                 ngf,
                 num_downs,
                 in_channels,
                 out_channels,
                 down_channels=(8, 8),
                 inter_channels=(8, 8),
                 up_channels=[[4, 8], [2, 4], [1, 2]],
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 loss_l1=dict(type='L1Loss'),
                 loss_vgg=dict(type='VGGLoss'),
                 loss_mask=dict(type='L1Loss'),
                 pretrained=None):
        super(Tryon, self).__init__()

        unet_block = builder.build_unet_skip_connection_block(
            dict(type='UnetSkipConnectionBlock',
                 outer_nc=ngf*down_channels[0],
                 inner_nc=ngf*down_channels[1],
                 input_nc=None,
                 submodule=None,
                 norm_layer=norm_layer,
                 innermost=True))

        for i in range(num_downs - 5):
            unet_block = builder.build_unet_skip_connection_block(
                dict(type='UnetSkipConnectionBlock',
                     outer_nc=ngf*inter_channels[0],
                     inner_nc=ngf*inter_channels[1],
                     input_nc=None,
                     submodule=unet_block,
                     norm_layer=norm_layer,
                     use_dropout=use_dropout))

        # upsample
        for ratio in up_channels:
            unet_block = builder.build_unet_skip_connection_block(
                dict(type='UnetSkipConnectionBlock',
                     outer_nc=ngf*ratio[0],
                     inner_nc=ngf*ratio[1],
                     input_nc=None,
                     submodule=unet_block,
                     norm_layer=norm_layer))

        unet_block = builder.build_unet_skip_connection_block(
            dict(type='UnetSkipConnectionBlock',
                 outer_nc=out_channels,
                 inner_nc=ngf,
                 input_nc=in_channels,
                 submodule=unet_block,
                 outermost=True,
                 norm_layer=norm_layer)
        )
        self.generator = unet_block

        self.loss_l1 = builder.build_loss(loss_l1)
        self.loss_vgg = builder.build_loss(loss_vgg)
        self.loss_mask = builder.build_loss(loss_mask)

        self.init_weights(pretrained=pretrained)

    def forward_train(self, img, agnostic, cloth, cloth_mask):
        input = torch.cat([agnostic, cloth], 1)
        output = self.generator(input)

        p_rendered, m_composite = torch.split(output, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = cloth * m_composite + p_rendered * (1 - m_composite)

        losses = dict()
        losses['loss_l1'] = self.loss_l1(p_tryon, img)
        losses['loss_vgg'] = self.loss_vgg(p_tryon, img)
        losses['loss_mask'] = self.loss_mask(m_composite, cloth_mask)

        return losses

    def forward_test(self, agnostic, cloth):
        input = torch.cat([agnostic, cloth], 1)
        output = self.generator(input)

        p_rendered, m_composite = torch.split(output, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = cloth * m_composite + p_rendered * (1 - m_composite)

        return p_tryon


    def forward(self,
                img,
                cloth,
                cloth_mask,
                agnostic,
                parse_cloth=None,
                im_name=None,
                c_name=None,
                return_loss=True):
        if return_loss:
            return self.forward_train(img, agnostic, cloth, cloth_mask)
        else:
            return self.forward_test(agnostic, cloth)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            self.unet.load_state_dict(torch.load(pretrained))