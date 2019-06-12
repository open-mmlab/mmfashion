from __future__ import division

import os
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

from mmcv.runner import Runner, DistSamplerSeedHook, obj_from_dict, save_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from .env import get_root_logger
from datasets import get_data, build_dataloader

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
           log_vars[loss_name] = loss.value.mean()
        elif isinstance(loss_value, list):
           log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
           raise TypeError(
                '{} is not a tensor or a list of tensors'.format(loss_name))

    loss = sum(_value for _key,_value in log_vars.items() if 'loss' in _key)
    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()
    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs

def train_predictor(model, dataset, cfg, distributed=False, validate=False, logger=None):
    if logger is None:
       logger = get_root_logger(cfg.log_level)

    # start training predictor
    if distributed: # to do
       _dist_train(model, dataset, cfg, validate=validate)
    else:
       _non_dist_train(model, dataset, cfg, validate=validate)

    
def build_optimizer(model, optim_cfg):
    if optim_cfg['type'] == 'SGD':
       optimizer = optim.SGD(model.parameters(), lr=optim_cfg.lr, momentum=optim_cfg.momentum)
    elif optim_cfg['type'] == 'Adam':
       optimizer = optim.Adam(model.parameters(), lr=optim_cfg.lr)
    return optimizer


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loader = build_dataloader(
                   dataset,
                   cfg.data.imgs_per_gpu,
                   cfg.data.workers_per_gpu,
                   cfg.gpus,
                   dist=False)
    
    print('dataloader built')

    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    print('model paralleled')
   
    criterion = torch.nn.BCELoss().cuda()
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    model.train()
    for epoch in range(cfg.start_epoch, cfg.end_epoch):
        if epoch%10==0:
           cfg.optimizer.lr = cfg.optimizer.lr*0.1
           optimizer = build_optimizer(model, cfg.optimizer)
       
        for batch_idx, traindata in enumerate(data_loader):
            imgs, target, landmarks, iuv = get_data(cfg, traindata)
            output = F.sigmoid(model(imgs, landmarks, iuv))
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % cfg.print_interval==0:
               print('Training Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.4f' %
                     (batch_idx, epoch, cfg.end_epoch, loss.item()))
            

        if epoch%cfg.save_interval==0:
           ckpt_path = os.path.join(cfg.work_dir, '%s_%s_epoch%d.pth.tar'%(cfg.arch, cfg.pooling,epoch))
           save_checkpoint(model, ckpt_path)
           print('Attribute Predictor saved in %s'% ckpt_path)

