import torch
import torch.nn as nn


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
    
    @property   
    def name(self):
        return self._name
   
    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class):
        ''' Register a module
        Args:
            module (:obj:'nn.Module'): Module to be registered.
        '''
        if not issubclass(module_class, nn.Module):
           raise TypeError(
                 'module must be a child of nn.Module, but got {}'.format(module_class))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls

LOSSES = Registry('loss') # loss function
BACKBONES = Registry('backbone') # basic feature extractor
GLOBALPOOLING = Registry('global_pool') # global pooling
ROIPOOLING = Registry('roi_pool') # roi pooling
CONCATS = Registry('concat') # concat local features and global features
PREDICTORS = Registry('predictor')
