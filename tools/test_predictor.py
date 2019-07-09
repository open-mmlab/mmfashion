from __future__ import division
import argparse

import torch
import torch.nn as nn

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.apis import (init_dist, get_root_logger, test_predictor)
from mmfashion.datasets.utils import get_dataset
from mmfashion.models import build_predictor
from mmfashion.utils import resume_from

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fashion Attribute Predictor')
    parser.add_argument('--config', help='train config file path', default='configs/RoI_Predictor.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--validate', action='store_true',
                         help='whether to evaluate the checkpoint during training', default=True)
    parser.add_argument('--gpus', type=int, default=4, help='number of gpus to use'
                                                 '(only applicable to non-distributed training)')
    parser.add_argument('--launcher',
                         choices=['none', 'pytorch','mpi','slurm'],
                         default='none',
                         help='job launcher')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
       cfg.work_dir = args.work_dir
    if args.resume_from is not None:
       cfg.resume_from = args.resume_from
    cfg.gpus.test = args.gpus

    # init distributed env first
    if args.launcher == 'none':
       distributed = False
    else:
       distributed = True
       init_dist(args.launcher, **cfg.dist_params)

    # init logger
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # data loader
    data_loader = get_dataset(cfg.data.test)
    print('dataset loaded')

    # build model and load checkpoint
    model = build_predictor(cfg.model)
    print('model built')
    model = resume_from(cfg, model)
        
    # test
    test_predictor(
                   model, 
                   data_loader, 
                   cfg, 
                   distributed=distributed, 
                   validate=args.validate,
                   logger=logger)

if __name__ == '__main__':
   main()
