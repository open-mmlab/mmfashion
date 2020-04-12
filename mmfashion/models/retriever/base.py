import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseRetriever(nn.Module):
    ''' Base class for fashion retriever'''
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseRetriever, self).__init__()

    @property
    def with_roi_pool(self):
        return hasattr(self, 'roi_pool') and self.roi_pool is not None

    @abstractmethod
    def simple_test(self, imgs, landmarks):
        pass

    @abstractmethod
    def aug_test(self, imgs, landmarks):
        pass

    def forward_test(self, imgs, landmarks):
        num_augs = len(imgs)
        if num_augs == 1:  # single image test
            return self.simple_test(imgs, landmarks)
        else:
            return self.aug_test(imgs, landmarks)

    @abstractmethod
    def forward_train(self, img, id, attr, pos, neg, anchor_lm, pos_lm, neg_lm,
                      triplet_pos_label, triplet_neg_label):
        pass

    def forward(self,
                img,
                landmark=None,
                id=None,
                attr=None,
                pos=None,
                neg=None,
                pos_lm=None,
                neg_lm=None,
                triplet_pos_label=None,
                triplet_neg_label=None,
                return_loss=True):
        if return_loss:
            return self.forward_train(img, id, attr, pos, neg, landmark,
                                      pos_lm, neg_lm, triplet_pos_label,
                                      triplet_neg_label)
        else:
            return self.forward_test(img, landmark)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
