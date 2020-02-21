from __future__ import division

import numpy as np
import torch
from mmcv.parallel import MMDataParallel

from ..core import Evaluator
from ..datasets import build_dataloader
from .env import get_root_logger


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


def _process_embeds(dataset, model, cfg):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    embeds = []
    with torch.no_grad():
        for data in data_loader:
            embed = model(data['img'], data['landmark'], return_loss=False)
            embeds.append(embed)

    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds


def _non_dist_test(model, query_set, gallery_set, cfg, validate=False):
    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()

    query_embeds = _process_embeds(query_set, model, cfg)
    gallery_embeds = _process_embeds(gallery_set, model, cfg)

    query_embeds_np = np.array(query_embeds)
    gallery_embeds_np = np.array(gallery_embeds)

    e = Evaluator(
        cfg.data.query.id_file,
        cfg.data.gallery.id_file,
        extract_feature=cfg.extract_feature)
    e.evaluate(query_embeds_np, gallery_embeds_np)


def _dist_test(model, query_set, gallery_set, cfg, validate=False):
    """ not implemented yet """
    raise NotImplementedError
