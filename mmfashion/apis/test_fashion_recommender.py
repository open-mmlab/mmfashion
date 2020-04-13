from __future__ import division

from mmcv.parallel import MMDataParallel
from ..datasets import build_dataloader
from .env import get_root_logger
import torch
import numpy as np


def test_fashion_recommender(model,
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
    embeddings = []
    for batch_idx, testdata in enumerate(data_loader):
        embed = model(testdata['img'], return_loss=False)
        embeddings.append(embed.data.cpu().numpy())

    # save as numpy array, and then transfer to tensor
    # this is to avoid out-of-memory
    embeddings = np.asarray(embeddings)
    embeddings = torch.from_numpy(embeddings)
    metric = model.triplet_net.metric_branch

    # compatibility auc
    auc = dataset.test_compatibility(embeddings, metric)

    # fill-in-blank accuracy
    acc = dataset.test_fitb(embeddings, metric)

    print('Compat AUC: {:.2f} FITB: {:.1f}\n'.format(
        round(auc, 2), round(acc * 100, 1)))


def _dist_test(model, dataset, cfg, validate=False):
    raise NotImplementedError
