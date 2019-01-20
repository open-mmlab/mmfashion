"""
Created on Wed Aug 01 2018
@author: Xin Liu, Ziwei Liu

Create a dataloader for AttrNet, which will process the dataset, load the data for training and testing
Note that the data is not the original images in DeepFashion, but the images cropped by their bboxes
"""

import argparse
import os, sys
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pdb

from data_processing import DataProcessing
from models.config import cfg


class data_loader():
    def __init__(self, BatchSize, NumWorkers):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformations = transforms.Compose([
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        dset_train = DataProcessing(cfg.IMG_PATH, cfg.TRAIN_DATA,
                                    cfg.TRAIN_IMG_FILE, cfg.TRAIN_LABEL_FILE,
                                    transformations, cfg.TRAIN_BBOX_FILE,
                                    cfg.LANDMARKS_TRAIN_FILE, cfg.IMG_SIZE)

        dset_test = DataProcessing(cfg.IMG_PATH, cfg.TEST_DATA,
                                   cfg.TEST_IMG_FILE, cfg.TEST_LABEL_FILE,
                                   transformations, cfg.TEST_BBOX_FILE,
                                   cfg.LANDMARKS_TEST_FILE, cfg.IMG_SIZE)

        # add bounding box to data loader
        self.train_loader = DataLoader(
            dset_train,
            batch_size=BatchSize,
            shuffle=True,
            num_workers=NumWorkers,
            pin_memory=True)
        self.test_loader = DataLoader(
            dset_test,
            batch_size=BatchSize,
            shuffle=False,
            num_workers=NumWorkers,
            pin_memory=True)
