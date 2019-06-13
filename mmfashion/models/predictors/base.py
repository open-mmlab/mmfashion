import logging 
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch
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
   def forward_train(self, img, landmarks=None, iuv=None):
       pass

   @abstractmethod
   def simple_test(self, imgs, landmarks, iuv):
       pass

   @abstractmethod
   def aug_test(self, imgs, landmarks, iuv):
       pass
 
   def forward_test(self, imgs, landmarks=None, iuv=None):
       num_augs = len(imgs)
       if num_augs == 1: # single image test
          return self.simple_test(imgs, landmarks, iuv)
       else:
          return self.aug_test(imgs, landmarks, iuv)

   @abstractmethod
   def forward_train(self, imgs, landmarks, iuv):
       pass

   def forward(self, img, landmarks=None, iuv=None, train=True):
       if train:
          return self.forward_train(img, landmarks, iuv)
       else:
          return self.forward_test(img, landmarks, iuv)


   def init_weights(self, pretrained=None):
       if pretrained is not None:
          logger = logging.getLogger()
          logger.info('load model from: {}'.format(pretrained))
