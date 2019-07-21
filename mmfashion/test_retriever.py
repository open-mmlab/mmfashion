from __future__ import division
import argparse

import torch
import torch.nn as nn

from mmcv import Config
from mmcv.runner import load_checkpoint

from apis import (init_dist, get_root_logger, test_retriever)
from datasets.utils import get_dataset
from models import build_retriever

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fashion Attribute Predictor')
    parser.add_argument('--config', help='train config file path', default='configs/roi_retriever_vgg.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/Predict/vgg/latest.pth', help='the checkpoint file to resume from')
    parser.add_argument('--validate', action='store_true',
                         help='whether to evaluate the checkpoint during training', default=True)
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
    query_set, gallery_set = get_dataset(cfg.data.query), get_dataset(cfg.data.gallery)
    print('dataset loaded')

    # build model and load checkpoint
    model = build_retriever(cfg.model)
    print('model built')
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        
    # test
    test_retriever(model, query_set, gallery_set, cfg, distributed=distributed, validate=args.validate, logger=logger)

if __name__ == '__main__':
   main()
