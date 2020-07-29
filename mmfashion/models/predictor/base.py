import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BasePredictor(nn.Module):
    ''' Base class for attribute predictors'''
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BasePredictor, self).__init__()

    @property
    def with_roi_pool(self):
        return hasattr(self, 'roi_pool') and self.roi_pool is not None

    @abstractmethod
    def simple_test(self, img, landmark):
        pass

    @abstractmethod
    def aug_test(self, img, landmark):
        pass

    def forward_test(self, img, landmark=None):
        num_augs = len(img)
        if num_augs == 1:  # single image test
            return self.simple_test(img[0], landmark[0])
        else:
            return self.aug_test(img, landmark)

    @abstractmethod
    def forward_train(self, img, landmark, attr, cate):
        pass

    def forward(self, img, attr=None, cate=None, landmark=None, return_loss=True):
        if return_loss:
            return self.forward_train(img, landmark, attr, cate)
        else:
            return self.forward_test(img, landmark)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
