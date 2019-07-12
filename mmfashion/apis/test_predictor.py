from __future__ import division

import os
import os.path as osp
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

from mmcv.runner import Runner, DistSamplerSeedHook, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from .env import get_root_logger
from .calculator import Calculator
from datasets import get_data, build_dataloader


def test_predictor(model, dataset, cfg, distributed=False, validate=False, logger=None):
    if logger is None:
       logger = get_root_logger(cfg.log_level)
    
    # start testing predictor
    if distributed: # to do 
       _dist_test(model, dataset, cfg, validate=validate)
    else:
      _non_dist_test(model, dataset, cfg, validate=validate)


def _non_dist_test(model, dataset, cfg, validate=False):
    data_loader = build_dataloader(
                   dataset,
                   cfg.data.imgs_per_gpu,
                   cfg.data.workers_per_gpu,
                   cfg.gpus.test,
                   dist=False,
                   shuffle=False)

    print('dataloader built')
 
    model = MMDataParallel(model, device_ids=range(cfg.gpus.test)).cuda()
    model.eval()
   
    #collector = build_collecter(cfg.class_num)
    calculator = Calculator(cfg.class_num)

    for batch_idx, testdata in enumerate(data_loader):
        imgs = testdata['img']
        landmarks = testdata['landmark']
        labels = testdata['label']
        
        predict = model(imgs, labels, landmarks, return_loss=False)        
        print('predict')
        print(predict.size())
        print(predict)
        calculator.collect_result(predict, labels)

        if batch_idx % cfg.print_interval == 0:
           calculator.show_result(batch_idx)
     
    calculator.show_result()               
 
