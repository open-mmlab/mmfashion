from __future__ import division
import argparse

from mmcv import Config

from mmfashion.apis import (get_root_logger, init_dist, set_random_seed,
                            train_predictor)
from mmfashion.datasets import get_dataset
from mmfashion.models import build_predictor
from mmfashion.utils import init_weights_from


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a Fashion Attribute Predictor')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/attribute_predict_coarse/roi_predictor_vgg_attr.py')
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
    model = build_predictor(cfg.model)
    print('model built')

    if cfg.init_weights_from:
        model = init_weights_from(cfg.init_weights_from, model)

    # data loader
    dataset = get_dataset(cfg.data.train)
    print('dataset loaded')

    # train
    train_predictor(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
