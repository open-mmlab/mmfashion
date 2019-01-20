from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.IMG_PATH = 'dataset/Img'

__C.DATA_PATH = 'dataset/labels'

# the path to save trained model
__C.saved_model_dir = 'saved_models'

# ImageNet pretrained vgg16 weights
__C.VGG16_BN_PATH = os.path.join(__C.saved_model_dir, 'vgg16_bn.pth')
__C.LANDMARKS_TRAIN_FILE = os.path.join(__C.DATA_PATH,
                                        'list_landmarks_train.txt')
__C.LANDMARKS_TEST_FILE = os.path.join(__C.DATA_PATH,
                                       'list_landmarks_test.txt')

__C.TRAIN_DATA = 'train'
__C.TEST_DATA = 'test'
__C.VAL_DATA = 'val'

# data path for training images and testing images
__C.TRAIN_IMG_FILE = os.path.join(__C.DATA_PATH, 'train.txt')
__C.TEST_IMG_FILE = os.path.join(__C.DATA_PATH, 'test.txt')
__C.VAL_IMG_FILE = os.path.join(__C.DATA_PATH, 'val.txt')

# data path for training labels and testing labels
__C.TRAIN_LABEL_FILE = os.path.join(__C.DATA_PATH, 'train_label.txt')
__C.TEST_LABEL_FILE = os.path.join(__C.DATA_PATH, 'test_label.txt')
__C.VAL_LABEL_FILE = os.path.join(__C.DATA_PATH, 'val_label.txt')

# data path for bounding box
__C.TRAIN_BBOX_FILE = os.path.join(__C.DATA_PATH, 'list_bbox_train.txt')
__C.TEST_BBOX_FILE = os.path.join(__C.DATA_PATH, 'list_bbox_test.txt')

__C.IMG_SIZE = (224, 224)

# the attribute file
__C.ATTR_FILE = os.path.join(__C.DATA_PATH, 'attributes.txt')

# the network architecture
__C.arch = 'vgg16'

# number of data loading workers
__C.num_workers = 6

# number of classes in the dataset
__C.num_classes = 88

__C.epochs = 60
# manual epoch number (useful on restarts)
__C.start_epoch = 0

# loss function, users can choose BCE or MSE
__C.loss = 'BCE'

__C.opt = 'SGD'

__C.batch_size = 64

# initial learning rate
__C.lr = 0.001

__C.momentum = 0.9

# if pretrained, change its value to 'False'
__C.pretrained = False

# Weight decay, for regularization
__C.weight_decay = 1e-4

parser = argparse.ArgumentParser(
    description='Train a network to recognize attribute.')

parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
global args
args = parser.parse_args()

# path to latest checkpoint (default: none)
__C.resume = args.resume
