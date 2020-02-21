from __future__ import division
import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.apis import get_root_logger, init_dist, test_landmark_detector
from mmfashion.datasets import get_dataset
from mmfashion.models import build_landmark_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fashion Landmark Detector Demo')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/landmark_detect/landmark_detect_vgg.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint/LandmarkDetect/vgg/global/landmark_detect_best.pth',
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training',
        default=True)
    parser.add_argument(
        '--use_cuda', type=bool, default=True, help='use gpu or not')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
