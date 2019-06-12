#import argparse
#import os
#import numpy as np

#from easydict import EasyDict
#cfg = EasyDict()

cfg.DataPath = 'datasets/dataset'
cfg.ImgPath = 'Img'
cfg.AnnoPath = 'Anno'
cfg.TrainImgFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'train_img.txt')
cfg.TrainLabelFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'train_labels.txt')
cfg.TrainBboxFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'train_bbox.txt') 
cfg.TrainLmFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'train_landmarks.txt')
cfg.TrainIUVFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'train_IUV.txt')

cfg.TestImgFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'test_img.txt')
cfg.TestLabelFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'test_labels.txt')
cfg.TestBboxFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'test_bbox.txt')
cfg.TestLmFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'test_landmarks.txt')
cfg.TestIUVFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'test_IUV.txt')

cfg.QueryImgFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'query_img.txt')
cfg.QueryLabelFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'query_labels.txt')
cfg.QueryBboxFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'query_bbox.txt')
cfg.QueryLmFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'query_landmarks.txt')
cfg.QueryIUVFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'query_IUV.txt')

cfg.GalleryImgFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'gallery_img.txt')
cfg.GalleryLabelFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'gallery_labels.txt')
cfg.GalleryBboxFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'gallery_bbox.txt')
cfg.GalleryLmFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'gallery_landmarks.txt')
cfg.GalleryIUVFile = os.path.join(cfg.DataPath, cfg.AnnoPath, 'gallery_IUV.txt')

cfg.width = 224
cfg.height=224
cfg.ImgSize=cfg.width, cfg.height

cfg.pretrained_weights='checkpoint/resnet50.pth'
cfg.save_dir = 'checkpoint'
cfg.pooling='IUV'

cfg.num_classes= 463
cfg.arch='vgg'
