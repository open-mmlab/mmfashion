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
   def forward(self, img, landmarks=None, iuv=None):
       pass

   def init_weights(self, pretrained=None):
       if pretrained is not None:
          logger = logging.getLogger()
          logger.info('load model from: {}'.format(pretrained))
