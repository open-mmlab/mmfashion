from __future__ import division
import argparse

import torch
import torch.nn as nn
import cv2
import numpy as np

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.datasets import build_dataset, build_dataloader
from mmfashion.models import build_retriever
from mmfashion.utils import img_to_tensor
from mmfashion.core import ClothesRetriever


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFashion In-shop Clothes Retriever Demo')
    parser.add_argument(
         '--input',
         type=str,
         help='input image path',
         default='demo/retrieve_demo1.jpg')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/retriever_in_shop/global_retriever_vgg_loss_id.py')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint/Retrieve/vgg/global/epoch_100.pth',
        help='the checkpoint file to resume from')    
    parser.add_argument(
         '--use_cuda',
          type=bool,
          default=True,
          help='use gpu or not')
    args = parser.parse_args()
    return args


def _process_embeds(dataset, model, cfg, use_cuda=True):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    total = 0
    embeds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if use_cuda:
               img = data['img'].cuda()
            embed = model(img, landmark=None, return_loss=False)
            embeds.append(embed)

    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    img = cv2.imread(args.input, -1)
    img = cv2.resize(img, (224,224))
    img_tensor = img_to_tensor(img, squeeze=True, cuda=args.use_cuda)

    model = build_retriever(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
 
    if args.use_cuda:
       model.cuda()

    model.eval()
   
    query_embed = model(img_tensor, landmark=None, return_loss=False)
   
    query_embed = query_embed.data.cpu().numpy()

    gallery_set = build_dataset(cfg.data.gallery)
    gallery_embeds = _process_embeds(gallery_set, model, cfg)

    retriever = ClothesRetriever(cfg.data.gallery.img_file, 
                                 cfg.data_root, 
                                 cfg.data.gallery.img_path)
    retriever.show_retrieved_images(query_embed, gallery_embeds)


if __name__ == '__main__':
   main()
