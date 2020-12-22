from __future__ import division
import argparse
import os

from mmcv import Config

from mmfashion.apis import (get_root_logger, init_dist, set_random_seed,
                            train_geometric_matching, train_tryon)
from mmfashion.datasets import get_dataset
from mmfashion.models import build_geometric_matching, build_tryon


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a Virtual Try-on Module')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/virtual_tryon/cp_vton.py')
    parser.add_argument(
        '--stage',
        required=True,
        help='train GMM(Geometric Matching Module) or TOM(Try-On Module)')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training',
        default=True)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'mpi', 'slurm'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    cfg.work_dir = os.path.join(cfg.work_dir, args.stage)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # init distributed env first
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # build model
    if args.stage == 'GMM':
        model = build_geometric_matching(cfg.GMM)
        print('Geometric Matching Module built')
        dataset = get_dataset(cfg.data.train.GMM)
        print('GMM dataset loaded')
        train_geometric_matching(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger)
    elif args.stage == 'TOM':
        model = build_tryon(cfg.TOM)
        print('Try-On Module built')
        dataset = get_dataset(cfg.data.train.TOM)
        print('TOM dataset loaded')
        train_tryon(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger)
    else:
        raise ValueError('stage should be GMM or TOM')


if __name__ == '__main__':
    main()
