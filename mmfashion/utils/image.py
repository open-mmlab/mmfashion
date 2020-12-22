from __future__ import division
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw


def get_img_tensor(img_path, use_cuda, get_size=False):
    img = Image.open(img_path)
    original_w, original_h = img.size

    img_size = (224, 224)  # crop image to (224, 224)
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
    if get_size:
        return img_tensor, original_w, original_w
    else:
        return img_tensor


def save_img(img_tensor, img_name):
    img_np = img_tensor.data.cpu().numpy().astype(np.uint8)
    if img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)

    elif img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)

    Image.fromarray(img_np).save(img_name)


def save_imgs(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255  # same with cp-vton
        tensor = tensor.cpu().clamp(0, 255)
        save_img(tensor, os.path.join(save_dir, img_name))


def show_img(img_tensor):
    plt.figure()
    img_np = img_tensor.data.cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    plt.imshow(img_np.transpose(1, 2, 0))
    plt.show()


def draw_landmarks(img_file, landmarks, r=2):
    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)
    for i, lm in enumerate(landmarks):
        x = lm[0]
        y = lm[1]
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(255, 0, 0, 0))
    img.show()
