import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(model, optim_cfg):
    if optim_cfg['type'] == 'SGD':
       optimizer = optim.SGD(model.parameters(), lr=optim_cfg.lr, momentum=optim_cfg.momentum)
    elif optim_cfg['type'] == 'Adam':
       optimizer = optim.Adam(model.parameters(), lr=optim_cfg.lr)
    return optimizer


def build_criterion(loss_dict):
    weight = loss_dict.weight
    size_average = loss_dict.size_average
    reduce = loss_dict.reduce
    reduction = loss_dict.reduction

    if loss_dict.type == 'CrossEntropyLoss':
       if loss_dict.use_sigmoid:
          return nn.BCEWithLogitsLoss(weight=weight,
                                      size_average=size_average,
                                      reduce=reduce,
                                      reduction=reduction)
       else:
          return nn.CrossEntropyLoss(weight=weight,
                                      size_average=size_average,
                                      reduce=reduce,
                                      reduction=reduction)
    else:
       raise TypeError('{} cannot be processed'.format(loss_dict.type))

