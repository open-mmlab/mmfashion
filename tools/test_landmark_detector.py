from __future__ import division
import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.apis import get_root_logger, init_dist, test_landmark_detector
from mmfashion.datasets import get_dataset
from mmfashion.models import build_landmark_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a Fashion Landmark Detector')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/landmark_detect/landmark_detect_resnet.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint/LandmarkDetect/vgg/latest.pth',
        help='the checkpoint file to resume from')
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

    # init distributed env first
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.checkpoint is not None:
        cfg.load_from = args.checkpoint

    # init logger
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed test: {}'.format(distributed))

    # data loader
    test_dataset = get_dataset(cfg.data.test)
    print('dataset loaded')

    # build model and load checkpoint
    model = build_landmark_detector(cfg.model)
    print('model built')

    load_checkpoint(model, cfg.load_from, map_location='cpu')
    print('load checkpoint from: {}'.format(cfg.load_from))

    # test
    test_landmark_detector(
        model,
        test_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
