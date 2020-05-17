from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from ..registry import ROIPOOLING


@ROIPOOLING.register_module
class RoIPooling(nn.Module):

    def __init__(self,
                 pool_plane,
                 inter_channels,
                 outchannels,
                 crop_size=7,
                 img_size=(224, 224),
                 num_lms=8,
                 roi_size=2):
        super(RoIPooling, self).__init__()
        self.maxpool = nn.MaxPool2d(pool_plane)
        self.linear = nn.Sequential(
            nn.Linear(num_lms * inter_channels, outchannels), nn.ReLU(True),
            nn.Dropout())

        self.inter_channels = inter_channels
        self.outchannels = outchannels
        self.num_lms = num_lms
        self.crop_size = crop_size
        assert img_size[0] == img_size[
            1], 'img width should equal to img height'
        self.img_size = img_size[0]
        self.roi_size = roi_size

        self.a = self.roi_size / float(self.crop_size)
        self.b = self.roi_size / float(self.crop_size)

    def forward(self, features, landmarks):
        """batch-wise RoI pooling.

        Args:
            features(tensor): the feature maps to be pooled.
            landmarks(tensor): crop the region of interest based on the
                landmarks(bs, self.num_lms).
        """
        batch_size = features.size(0)

        # transfer landmark coordinates from original image to feature map
        landmarks = landmarks / self.img_size * self.crop_size
        landmarks = landmarks.view(batch_size, self.num_lms, 2)

        ab = [np.array([[self.a, 0], [0, self.b]]) for _ in range(batch_size)]
        ab = np.stack(ab, axis=0)
        ab = torch.from_numpy(ab).float().cuda()
        size = torch.Size(
            (batch_size, features.size(1), self.roi_size, self.roi_size))

        pooled = []
        for i in range(self.num_lms):
            tx = -1 + 2 * landmarks[:, i, 0] / float(self.crop_size)
            ty = -1 + 2 * landmarks[:, i, 1] / float(self.crop_size)
            t_xy = torch.stack((tx, ty)).view(batch_size, 2, 1)
            theta = torch.cat((ab, t_xy), 2)

            flowfield = nn.functional.affine_grid(theta, size)
            one_pooled = nn.functional.grid_sample(
                features,
                flowfield.to(torch.float32),
                mode='bilinear',
                padding_mode='border')
            one_pooled = self.maxpool(one_pooled).view(batch_size,
                                                       self.inter_channels)

            pooled.append(one_pooled)
        pooled = torch.stack(pooled, dim=1).view(batch_size, -1)
        pooled = self.linear(pooled)
        return pooled

    def init_weights(self):
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
