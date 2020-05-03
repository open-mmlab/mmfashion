from __future__ import division

import torch
from mmcv.parallel import MMDataParallel

from ..datasets import build_dataloader
from .env import get_root_logger


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


def _process_embeds(dataset, model, cfg):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)
    print('dataloader built')

    embeds = []
    with torch.no_grad():
        for data in data_loader:
            embed = model(data['img'], return_loss=False)
            embeds.append(embed.data.cpu())

    embeds = torch.cat(embeds)
    return embeds


def _non_dist_test(model, dataset, cfg, validate=False):
    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()

    embeds = _process_embeds(dataset, model, cfg)

    metric = model.module.triplet_net.metric_branch

    # compatibility auc
    auc = dataset.test_compatibility(embeds, metric)

    # fill-in-blank accuracy
    acc = dataset.test_fitb(embeds, metric)

    print('Compat AUC: {:.2f} FITB: {:.1f}\n'.format(
        round(auc, 2), round(acc * 100, 1)))


def _dist_test(model, dataset, cfg, validate=False):
    raise NotImplementedError
