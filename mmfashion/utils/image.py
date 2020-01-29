from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch


def img_to_tensor(img, squeeze=False, cuda=False):
    """ transform cv2 read numpy array to torch tensor
    Args:
    img(numpy arrary): cv2 read img, [H,W,C]
    """

    img = (img.astype(np.float32))[:, :, ::-1]  # bgr to rgb
    img_norm = np.clip(img / 255., 0, 1)
    img_norm = img_norm.transpose(2, 0, 1)  # [h,w,c] to [c,h,w]

    # transfer to tensor
    img_tensor = torch.from_numpy(img_norm)

    # add one dimension
    if squeeze:
        img_tensor = img_tensor.unsqueeze(0)
    if cuda:
        img_tensor = img_tensor.cuda()
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
