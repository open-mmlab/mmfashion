from __future__ import division
import argparse
import os

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.apis import get_root_logger
from mmfashion.datasets import build_dataset
from mmfashion.models import build_fashion_recommender
from mmfashion.utils import get_img_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a Fashion Attribute Predictor')
    parser.add_argument(
        '--config',
        help='test config file path',
        default='configs/fashion_recommendation/'
        'type_aware_recommendation_polyvore_disjoint_l2_embed.py')
    parser.add_argument(
        '--checkpoint',
        help='checkpoint file',
        default='checkpoint/FashionRecommend/TypeAware/disjoint/'
        'l2_embed/epoch_16.pth')
    parser.add_argument(
        '--input_dir',
        type=str,
        help='input image path',
        default='demo/imgs/fashion_compatibility/set2')
    parser.add_argument(
        '--use_cuda', type=bool, default=True, help='use gpu or not')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.load_from = args.checkpoint
    # init distributed env first
    distributed = False

    # init logger
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed test: {}'.format(distributed))

    # create model
    model = build_fashion_recommender(cfg.model)
    load_checkpoint(model, cfg.load_from, map_location='cpu')
    print('load checkpoint from: {}'.format(cfg.load_from))
    if args.use_cuda:
        model.cuda()
    model.eval()

    # prepare input data
    img_tensors = []
    item_ids = []

    for dirpath, dirname, fns in os.walk(args.input_dir):
        for imgname in fns:
            item_ids.append(imgname.split('.')[0])
            tensor = get_img_tensor(
                os.path.join(dirpath, imgname), args.use_cuda)
            img_tensors.append(tensor)
    img_tensors = torch.cat(img_tensors)

    # test
    embeds = []
    with torch.no_grad():
        embed = model(img_tensors, return_loss=False)
        embeds.append(embed.data.cpu())
    embeds = torch.cat(embeds)

    try:
        metric = model.module.triplet_net.metric_branch
    except Exception:
        metric = None

    # get compatibility score, so far only support images from polyvore
    dataset = build_dataset(cfg.data.test)

    score = dataset.get_single_compatibility_score(embeds, item_ids, metric,
                                                   args.use_cuda)
    print("Compatibility score: {:.3f}".format(score))


if __name__ == '__main__':
    main()
