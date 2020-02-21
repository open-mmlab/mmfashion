from __future__ import division
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import ClothesRetriever
from mmfashion.datasets import build_dataloader, build_dataset
from mmfashion.models import build_retriever


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFashion In-shop Clothes Retriever Demo')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='demo/imgs/06_1_front.jpg')
    parser.add_argument(
        '--topk',
        type=int,
        default=5,
        help='retrieve topk items'
    )
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
            embed = model(img, landmark= data['landmark'], return_loss=False)
            embeds.append(embed)

    embeds = torch.cat(embeds)
    embeds = embeds.data.cpu().numpy()
    return embeds


def get_img_tensor(img_path, use_cuda):
    img = Image.open(img_path)
    img_size = (224,224)
    img.thumbnail(img_size, Image.ANTIALIAS)
    img = img.convert('RGB')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    if use_cuda:
        img_tensor = img_tensor.cuda()
    return img_tensor


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

    query_embed = model(img_tensor, landmark=None, return_loss=False)
    query_embed = query_embed.data.cpu().numpy()

    gallery_set = build_dataset(cfg.data.gallery)
    gallery_embeds = _process_embeds(gallery_set, model, cfg)

    retriever = ClothesRetriever(cfg.data.gallery.img_file, cfg.data_root,
                                 cfg.data.gallery.img_path)
    retriever.show_retrieved_images(query_embed, gallery_embeds)



if __name__ == '__main__':
    main()
