from __future__ import division

import os
import os.path as osp
import re
from collections import OrderedDict
from scipy.spatial.distance import cdist
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
from datasets import get_data, build_dataloader
from utils import save_checkpoint, resume_from


def test_retriever(model, query_set, gallery_set, cfg, distributed=False, validate=False, logger=None):
    if logger is None:
       logger = get_root_logger(cfg.log_level)
    
    # start testing predictor
    if distributed: # to do 
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
                   cfg.gpus.test,
                   dist=False,
                   shuffle=False)
    
    embeds = []
    for batch_idx, data in enumerate(data_loader):
        embed = model(data['img'],
                      data['landmark'],
                      return_loss=False)
        embeds.append(embed.data.cpu().numpy())
    return embeds


def _calculate(idxes, idx2id, query_id):
    ids = []
    cnt = 0
    for i in idxes:
        ids.append(idx2id[i])
    if query_id in ids:
       cnt += 1
    return cnt 

def show_result(query_embeds, gallery_embeds, query_dict, gallery_dict):
    top1, top3, top5, top10 = 0, 0, 0,0
    total = len(query_embeds)

    for qi, query_embed in enumerate(query_embeds):
        dist = []
        for gi, gallery_embed in enumerate(gallery_embeds):
            one_dist = cdist(query_embed, gallery_embed, 'euclidean')
            dist.append(one_dist[0][0])
        dist = np.asarray(dist)
                 
        order = np.argsort(dist)
        query_id = query_dict[qi]

        top1 += _calculate(order[:1], gallery_dict, query_id)
        top3 += _calculate(order[:3], gallery_dict, query_id)
        top5 += _calculate(order[:5], gallery_dict, query_id)
        top10 += _calculate(order[:10], gallery_dict, query_id)
        print(top1, top3, top5, top10)
        acc1, acc3, acc5, acc10 = 100*float(top1)/ total, 100*float(top3)/ total, 100*float(top5)/ total, 100*float(top10)/ total
        print('top1 = %.4f, top3 = %.4f, top5 = %.4f, top10 = %.4f '%
              (acc1, acc3, acc5, acc10))

    print('------------- Recall Rate ------------------')
    print(top1, top3, top5, top10)
    acc1, acc3, acc5, acc10 = 100*float(top1)/ total, 100*float(top3)/ total, 100*float(top5)/ total, 100*float(top10)/ total
    print('top1 = %.4f, top3 = %.4f, top5 = %.4f, top10 = %.4f '%
            (acc1, acc3, acc5, acc10))

def _non_dist_test(model, query_set, gallery_set, cfg, validate=False):

    model = MMDataParallel(model, device_ids=range(cfg.gpus.test)).cuda()
    model.eval()

    query_embeds = _process_embeds(query_set, model, cfg)
    gallery_embeds = _process_embeds(gallery_set, model, cfg)

    show_result(query_embeds, gallery_embeds, query_set.idx2id, gallery_set.idx2id)

