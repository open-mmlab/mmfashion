from __future__ import division
import argparse
import os

import scipy.io as sio
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmfashion.datasets import build_dataloader, build_dataset
from mmfashion.models import build_retriever


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a Fashion Attribute Predictor')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/retriever/roi_retriever_vgg.py')
    parser.add_argument(
        '--data_type',
        type=str,
        default='train',
        help='extract features from train/query/gallery list')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='extracted_features',
        help='directory to save extracted features')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint/Retrieve/vgg/roi/latest.pth',
        help='the checkpoint file to resume from')
    args = parser.parse_args()
    return args


def _process_embeds(dataset, model, cfg):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)
    embeds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            embed = model(data['img'], data['landmark'], return_loss=False)
            embeds.append(embed)
    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds


def extract_features(image_set, cfg, save_feature_dir):

    model = build_retriever(cfg.model)
    print('model built')
    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()

    embeds = _process_embeds(image_set, model, cfg)

    if not os.path.exists(save_feature_dir):
        os.makedirs(save_feature_dir)
    save_path = os.path.join(save_feature_dir, 'extracted_features.mat')

    sio.savemat(save_path, {'embeds': embeds})
    print('extracted features saved to : %s' % save_path)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.data_type == 'train':
        image_set = build_dataset(cfg.data.train)
    elif args.data_type == 'query':
        image_set = build_dataset(cfg.data.query)
    elif args.data_type == 'gallery':
        image_set = build_dataset(cfg.data.gallery)
    else:
        raise TypeError('So far only support train/query/gallery dataset')

    if args.checkpoint is not None:
        cfg.load_from = args.checkpoint

    extract_features(image_set, cfg, args.save_dir)


if __name__ == '__main__':
    main()
