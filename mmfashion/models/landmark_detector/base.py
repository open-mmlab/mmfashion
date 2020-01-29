import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseLandmarkDetector(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseLandmarkDetector, self).__init__()

    @abstractmethod
    def simple_test(self, img, landmark):
        pass

    @abstractmethod
    def aug_test(self, img, landmark):
        pass

    def forward_test(self, img):
        num_augs = len(img)
        if num_augs == 1:  # single image test
            return self.simple_test(img[0])
        else:
            return self.aug_test(img)

    @abstractmethod
    def forward_train(self, img, vis, landmark_for_regreesion,
                      landmark_for_roi_pool, attr):
        pass

    def forward(self,
                img,
                vis=None,
                landmark_for_regression=None,
                landmark_for_roi_pool=None,
                attr=None,
                return_loss=True):
        if return_loss:
            return self.forward_train(img, vis, landmark_for_regression,
                                      landmark_for_roi_pool, attr)
        else:
            return self.forward_test(img)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
