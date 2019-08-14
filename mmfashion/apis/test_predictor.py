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
from core import AttrCalculator, CateCalculator
from datasets import build_dataloader


def test_predictor(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start testing predictor
    if distributed:  # to do
        _dist_test(model, dataset, cfg, validate=validate)
    else:
        _non_dist_test(model, dataset, cfg, validate=validate)


def _non_dist_test(model, dataset, cfg, validate=False):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    print('dataloader built')

    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()

    attr_calculator = AttrCalculator(cfg)
    cate_calculator = CateCalculator(cfg)

    for batch_idx, testdata in enumerate(data_loader):
        imgs = testdata['img']
        landmarks = testdata['landmark']
        attr = testdata['attr']
        cate = testdata['cate']

        attr_pred, cate_pred = model(imgs, attr, cate, landmarks, return_loss=False)
        attr_calculator.collect_result(attr_pred, attr)
        cate_calculator.collect_result(cate_pred, cate)

        if batch_idx % cfg.print_interval == 0:
            attr_calculator.show_result(batch_idx)
            cate_calculator.show_result()

    attr_calculator.show_result()
    cate_calculator.show_result()
