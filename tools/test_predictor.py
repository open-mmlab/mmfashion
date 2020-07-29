from __future__ import division
import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.apis import get_root_logger, init_dist, test_predictor
from mmfashion.datasets.utils import get_dataset
from mmfashion.models import build_predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a Fashion Attribute Predictor')
    parser.add_argument(
        '--config',
        help='test config file path',
        default='configs/attribute_predict_coarse/roi_predictor_vgg.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--checkpoint',
        help='checkpoint file',
        default='checkpoint/Predict/vgg/roi/latest.pth')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training',
        default=True)
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

    # data loader
    dataset = get_dataset(cfg.data.test)
    print('dataset loaded')

    # build model and load checkpoint
    model = build_predictor(cfg.model)
    print('model built')

    load_checkpoint(model, cfg.load_from, map_location='cpu')
    print('load checkpoint from: {}'.format(cfg.load_from))

    # test
    test_predictor(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
