from __future__ import division
import argparse

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.models import build_landmark_detector
from mmfashion.utils import get_img_tensor, draw_landmarks


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fashion Landmark Detector Demo')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='demo/imgs/landmark_predict/demo1.jpg')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/landmark_detect/landmark_detect_vgg.py')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint/LandmarkDetect/global/landmark_detect_best.pth',
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
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = parse_args()
    cfg = Config.fromfile(args.config)

    img_tensor, w, h = get_img_tensor(args.input, args.use_cuda, get_size=True)

    # build model and load checkpoint
    model = build_landmark_detector(cfg.model)
    print('model built')
    load_checkpoint(model, args.checkpoint)
    print('load checkpoint from: {}'.format(args.checkpoint))

    if args.use_cuda:
        model.cuda()

    # detect landmark
    model.eval()
    pred_vis, pred_lm = model(img_tensor, return_loss=False)
    pred_lm = pred_lm.data.cpu().numpy()
    vis_lms = []

    for i, vis in enumerate(pred_vis):
        if vis >= 0.5:
            print('detected landmark {} {}'.format(
                pred_lm[i][0] * (w / 224.), pred_lm[i][1] * (h / 224.)))
            vis_lms.append(pred_lm[i])

    draw_landmarks(args.input, vis_lms)



if __name__ == '__main__':
    main()
