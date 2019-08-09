from __future__ import division

import os
import os.path as osp
import re
from collections import OrderedDict
from scipy.spatial.distance import cdist
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

from mmcv.runner import Runner, DistSamplerSeedHook, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from .env import get_root_logger
from core import Evaluator
from datasets import build_dataloader


def test_retriever(model,
                   query_set,
                   gallery_set,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start testing predictor
    if distributed:  # to do
        _dist_test(model, query_set, gallery_set, cfg, validate=validate)
    else:
        _non_dist_test(model, query_set, gallery_set, cfg, validate=validate)


def _dist_test(model, query_set, gallery_set, cfg, validate=False):
    """ not implemented yet """
    raise NotImplementedError


def _process_embeds(dataset, model, cfg):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    total = 0
    embeds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            embed = model(data['img'], data['landmark'], return_loss=False)
            embeds.append(torch.sigmoid(embed))

    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds


def _non_dist_test(model, query_set, gallery_set, cfg, validate=False):

    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()

    query_embeds = _process_embeds(query_set, model, cfg)
    gallery_embeds = _process_embeds(gallery_set, model, cfg)

    query_embeds_np = np.array(query_embeds)
    print('query_embeds', query_embeds_np.shape)
    sio.savemat('query_embeds.mat', {'embeds': query_embeds_np})

    gallery_embeds_np = np.array(gallery_embeds)
    print('gallery_embeds', gallery_embeds_np.shape)
    sio.savemat('gallery_embeds.mat', {'embeds': gallery_embeds_np})

    e = Evaluator(cfg.data.query.idx2id, cfg.data.gallery.idx2id)
    e.evaluate(query_embeds_np, gallery_embeds_np)
