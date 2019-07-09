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
   def forward_train(self, img, landmarks=None):
       pass

   @abstractmethod
   def simple_test(self, imgs, landmarks):
       pass

   @abstractmethod
   def aug_test(self, imgs, landmarks):
       pass
 
   def forward_test(self, imgs, landmarks=None):
       num_augs = len(imgs)
       if num_augs == 1: # single image test
          return self.simple_test(imgs[0], landmarks[0])
       else:
          return self.aug_test(imgs, landmarks)

   @abstractmethod
   def forward_train(self, imgs, labels, landmarks):
       pass

   def forward(self, img, label, landmarks=None, return_loss=True):
       if return_loss:
          return self.forward_train(img, label, landmarks)
       else:
          return self.forward_test(img, landmarks)


   def init_weights(self, pretrained=None):
       if pretrained is not None:
          logger = logging.getLogger()
          logger.info('load model from: {}'.format(pretrained))
