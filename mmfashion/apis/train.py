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

from mmcv.runner import Runner, DistSamplerSeedHook, obj_from_dict, save_checkpoint, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from .env import get_root_logger
from .utils import build_optimizer, build_criterion
from datasets import get_data, build_dataloader



def train_predictor(model, dataset, cfg, distributed=False, validate=False, logger=None):
    if logger is None:
       logger = get_root_logger(cfg.log_level)

    # start training predictor
    if distributed: # to do
       _dist_train(model, dataset, cfg, validate=validate)
    else:
       _non_dist_train(model, dataset, cfg, validate=validate)

    

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
   
    optimizer = build_optimizer(model, cfg.optimizer)
    criterion = build_criterion(cfg.loss_dict)
        
    if cfg.load_from:
       checkpoint = load_checkpoint(model, cfg.load_from)
       print('load checkpoint: {}'.format(cfg.load_from))
   
    model.train()
    for epoch in range(cfg.start_epoch, cfg.end_epoch):
        if epoch%cfg.lr_config.warmup_iters==0:
           cfg.optimizer.lr = cfg.optimizer.lr * cfg.lr_config.warmup_ratio
           optimizer = build_optimizer(model, cfg.optimizer)

        for batch_idx, traindata in enumerate(data_loader):
            imgs, labels, landmarks, iuv = get_data(cfg, traindata)
         
            pred = model(imgs, landmarks, iuv, train=True)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            if batch_idx%cfg.print_interval==0:
               print('Training Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.4f' %
                     (batch_idx, epoch, cfg.end_epoch, loss.item()))
            

        if epoch%cfg.save_interval==0:
           ckpt_path = os.path.join(cfg.work_dir, '%s_%s_epoch%d.pth.tar'%(cfg.arch, cfg.pooling,epoch))
           save_checkpoint(model, ckpt_path)
           print('Attribute Predictor saved in %s'% ckpt_path)
