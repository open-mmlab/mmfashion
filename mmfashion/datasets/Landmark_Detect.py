from __future__ import division
from functools import partial

import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mmcv.runner import get_dist_info
from mmcv.parallel import collate

import os
import sys
import random
from skimage import io
from PIL import Image
import numpy as np

from .loader import GroupSampler, DistributedGroupSampler, DistributedSampler
from .registry import DATASETS

@DATASETS.register_module
class LandmarkDetectDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 img_path,
                 img_file,
                 bbox_file,
                 landmark_file,
                 img_size,
                 attr_file=None):
        self.img_path = img_path
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # read img names
        fp = open(img_file, 'r')
        self.img_list = [x.strip() for x in fp]

        self.img_size = img_size
        # load bbox
        if bbox_file:
            self.with_bbox = True
            self.bboxes = np.loadtxt(bbox_file, usecols=(0, 1, 2, 3))
        else:
            self.with_bbox = False
            self.bboxes = None

        # load landmarks and visibility
        self.landmarks = np.loadtxt(landmark_file, dtype=np.float)
        
        # load attributes
        if attr_file is not None:
           self.attributes = np.loadtxt(attr_file, dtype=np.float)
        else:
           self.attributes = None


    def get_basic_item(self, idx):
        img = Image.open(os.path.join(self.img_path, self.img_list[idx])).convert('RGB')

        width, height = img.size
        if self.with_bbox:
            bbox_cor = self.bboxes[idx]
            x1 = max(0, int(bbox_cor[0]) - 10)
            y1 = max(0, int(bbox_cor[1]) - 10)
            x2 = int(bbox_cor[2]) + 10
            y2 = int(bbox_cor[3]) + 10
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            img = img.crop(box=(x1, y1, x2, y2))
        else:
            bbox_w, bbox_h = self.img_size[0], self.img_size[1]

        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = self.transform(img)
        
        landmark, vis = [], []
        origin_landmark = self.landmarks[idx]
        
        # compute the shiftness
        for i, l in enumerate(origin_landmark):
            if i%3 == 0: # visibility
               vis.append(l)
            else:
               if i%3==1: # x
                  l_x = max(0, l-x1)
                  l_x = float(l_x) / bbox_w * self.img_size[0]
                  landmark.append(l_x)
               else: # y
                  l_y = max(0, l-y1)
                  l_y = float(l_y) / bbox_h * self.img_size[1]
                  landmark.append(l_y)
        landmark = torch.from_numpy(np.array(landmark)).float()
        vis = torch.from_numpy(np.array(vis)).float()
        
        # load attribute
        if self.attributes is not None:
           attr = self.attributes[idx]
        else:
           attr = None

        data = {'img': img,
                'landmark': landmark,
                 'vis': vis}
        return data

    def __getitem__(self, idx):
        return self.get_basic_item(idx)

    def __len__(self):
        return len(self.img_list)
