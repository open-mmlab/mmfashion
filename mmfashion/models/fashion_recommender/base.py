import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseFashionRecommender(nn.Module):
    ''' Base class for fashion recommender'''
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseFashionRecommender, self).__init__()

    @abstractmethod
    def forward_test(self, imgs):
        pass

    @abstractmethod
    def forward_train(self, img, text, has_text, pos_img, pos_text,
                      pos_has_text, neg_img, neg_text, neg_has_text,
                      condition):
        pass

    def forward(self,
                img,
                text=None,
                has_text=None,
                pos_img=None,
                pos_text=None,
                pos_has_text=None,
                neg_img=None,
                neg_text=None,
                neg_has_text=None,
                condition=None,
                return_loss=True):
        if return_loss:
            return self.forward_train(img, text, has_text, pos_img, pos_text,
                                      pos_has_text, neg_img, neg_text,
                                      neg_has_text, condition)
        else:
            return self.forward_test(img)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
