from __future__ import division

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
    img_np = img_tensor.data.cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8).transpose(1, 2, 0)
    matplotlib.image.imsave(img_name, img_np)


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
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255, 0, 0, 0))
    img.show()
