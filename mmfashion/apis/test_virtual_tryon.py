from __future__ import division

import os
import numpy as np
import torch

from ..utils import save_imgs
from ..datasets import build_dataloader
from .env import get_root_logger


def test_geometric_matching(model,
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
        _non_dist_test_gmm(model, dataset, cfg, validate=validate)


def _non_dist_test_gmm(model, dataset, cfg, validate=False):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    # save dir
    warp_cloth_dir = os.path.join(cfg.data.test.GMM.save_dir, 'warp-cloth')
    warp_mask_dir = os.path.join(cfg.data.test.GMM.save_dir, 'warp-mask')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    model.cuda()
    model.eval()

    for batch, data in enumerate(data_loader):
        c_name = data['c_name']
        cloth = data['cloth'].cuda()
        cloth_mask = data['cloth_mask'].cuda()
        agnostic = data['agnostic'].cuda()
        parse_cloth = data['parse_cloth'].cuda()

        warped_cloth, warped_mask = model(cloth,
                                          cloth_mask,
                                          agnostic,
                                          parse_cloth,
                                          return_loss=False)
        save_imgs(warped_cloth, c_name, warp_cloth_dir)
        save_imgs(warped_mask, c_name, warp_mask_dir)

def test_tryon(model,
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
        _non_dist_test_tryon(model, dataset, cfg, validate=validate)

def _non_dist_test_tryon(model, dataset, cfg, validate=False):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    # save dir
    try_on_dir = os.path.join(cfg.data.test.TOM.save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)

    model.cuda()
    model.eval()

    for batch, data in enumerate(data_loader):
        img = data['img'].cuda()
        cloth = data['cloth'].cuda()
        cloth_mask = data['cloth_mask'].cuda()
        agnostic = data['agnostic'].cuda()
        im_names = data['im_name']

        p_tryon = model(img,
                        cloth,
                        cloth_mask,
                        agnostic,
                        return_loss=False)

        save_imgs(p_tryon, im_names, try_on_dir)


def _dist_test(model, dataset, cfg, validate=False):
    """ not implemented yet """
    raise NotImplementedError
