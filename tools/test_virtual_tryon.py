from __future__ import division
import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.apis import (get_root_logger, init_dist,
                            test_geometric_matching, test_tryon)
from mmfashion.datasets.utils import get_dataset
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
        help='test GMM(Geometric Matching Module) or TOM(Try-On Module)')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--checkpoint',
        help='checkpoint file',
        default='checkpoint/CPVTON/TOM/latest.pth')
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
    if args.checkpoint is not None:
        cfg.load_from = args.checkpoint
    # init distributed env first
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed test: {}'.format(distributed))

    # build model and load checkpoint
    if args.stage == 'GMM':
        # test geometric matching
        # data loader
        dataset = get_dataset(cfg.data.test.GMM)
        print('GMM dataset loaded')

        model = build_geometric_matching(cfg.GMM)
        print('GMM model built')
        load_checkpoint(model, cfg.load_from, map_location='cpu')
        print('load checkpoint from: {}'.format(cfg.load_from))

        test_geometric_matching(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=False,
            logger=logger)

    elif args.stage == 'TOM':
        # test tryon module
        dataset = get_dataset(cfg.data.test.TOM)
        print('TOM dataset loaded')

        model = build_tryon(cfg.TOM)
        print('TOM model built')
        load_checkpoint(model, cfg.load_from, map_location='cpu')
        print('load checkpoint from: {}'.format(cfg.load_from))

        test_tryon(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=False,
            logger=logger)


if __name__ == '__main__':
    main()
