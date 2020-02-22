from __future__ import division
import argparse

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import ClothesRetriever
from mmfashion.datasets import build_dataloader, build_dataset
from mmfashion.models import build_retriever
from mmfashion.utils import get_img_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFashion In-shop Clothes Retriever Demo')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='demo/imgs/06_1_front.jpg')
    parser.add_argument(
        '--topk', type=int, default=5, help='retrieve topk items')
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
        '--use_cuda', type=bool, default=True, help='use gpu or not')
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

    embeds = []
    with torch.no_grad():
        for data in data_loader:
            if use_cuda:
                img = data['img'].cuda()
            embed = model(img, landmark=data['landmark'], return_loss=False)
            embeds.append(embed)

    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds


def main():
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = parse_args()
    cfg = Config.fromfile(args.config)

    model = build_retriever(cfg.model)
    load_checkpoint(model, args.checkpoint)
    print('load checkpoint from {}'.format(args.checkpoint))

    if args.use_cuda:
        model.cuda()
    model.eval()

    img_tensor = get_img_tensor(args.input, args.use_cuda)

    query_feat = model(img_tensor, landmark=None, return_loss=False)
    query_feat = query_feat.data.cpu().numpy()

    gallery_set = build_dataset(cfg.data.gallery)
    gallery_embeds = _process_embeds(gallery_set, model, cfg)

    retriever = ClothesRetriever(cfg.data.gallery.img_file, cfg.data_root,
                                 cfg.data.gallery.img_path)
    retriever.show_retrieved_images(query_feat, gallery_embeds)


if __name__ == '__main__':
    main()
