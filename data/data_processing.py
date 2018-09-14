"""
Created on Wed Aug 01 2018
@author: Xin Liu, Ziwei Liu

Create a DataProcessing class here, which is aimed to crop the region based on its bbox, and then to resize images
"""


import torch
import os
import sys
from skimage import io
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import models.config as cfg


class DataProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filename,label_filename, transform, bbox_file, landmarks_file, img_size):

        self.data_path = data_path
        self.img_path = os.path.join(data_path,img_path)
        self.transform = transform

        # read img file from file ('train.txt', 'val.txt','test.txt')
        img_filepath = img_filename
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()

        #read labels from file
        label_filepath = label_filename
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        
        self.label = labels

        # load the whole bboxes as matrix
        self.bbox_file = bbox_file

        self.bboxes = np.loadtxt(bbox_file, usecols=(0,1,2,3))

        # load the rois
        
        self.landmarks = np.loadtxt(landmarks_file)
        
        self.img_size = img_size


    # read img, select bounding_box region, and read its label
    def __getitem__(self,idx):
        
        # crop the bbox region
        bbox_cor = self.bboxes[idx]

        x1 = int(bbox_cor[0])-10
        y1 = int(bbox_cor[1])-10
        x2 = int(bbox_cor[2])+10
        y2 = int(bbox_cor[3])+10
        
        # crop the image
        img = Image.open(os.path.join(self.data_path,self.img_filename[idx])).crop(box=(x1,y1,x2,y2))
        
        # resize img
        img.thumbnail(self.img_size, Image.ANTIALIAS)
        
        img = img.convert('RGB') # TBD

        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.label[idx])
        landmarks = torch.from_numpy(self.landmarks[idx])
        return img, label, landmarks

        
    def __len__(self):
        return len(self.img_filename)
