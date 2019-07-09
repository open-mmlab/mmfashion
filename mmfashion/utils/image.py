from __future__ import division

import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torchvision

def save_img(img_tensor, img_name):
    img_np = img_tensor.data.cpu().numpy()
    img_np = (img_np*255).astype(np.uint8).transpose(1,2,0)

    matplotlib.image.imsave(img_name, img_np)

def show_img(img_tensor):
    plt.figure()
    img_np = img_tensor.data.cpu().numpy()
    img_np = (img_np*255).astype(np.uint8)
    plt.imshow(img_np.transpose(1,2,0))
    plt.show()


