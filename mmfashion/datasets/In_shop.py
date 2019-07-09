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


class InShopDataset(Dataset):
    def __init__(self, img_path, img_file, label_file, bbox_file, landmark_file, img_size, find_three=False):
       self.img_path = img_path
       
       normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
       self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
       ])
 
       # read img names
       fp = open(img_file, 'r')
       self.img_list = [x.strip() for x in fp]

       # collect id
       self.id2idx, self.idx2id = {},{}
       self.ids = []
       for idx, img_name in enumerate(self.img_list):
           img_id = int(img_name.split('/')[3].split('_')[1])
           self.idx2id[idx] = img_id
           if img_id not in self.id2idx:
              self.id2idx[img_id] = [idx]
              self.ids.append(img_id)
           else:
              self.id2idx[img_id].append(idx)
       fp.close()

       # read labels
       self.labels = np.loadtxt(label_file, dtype=np.float32)
       
       self.img_size = img_size
       
       # load bbox
       if bbox_file:
          self.with_bbox = True
          self.bboxes = np.loadtxt(bbox_file, usecols=(0,1,2,3))
       else:
          self.with_bbox = False
          self.bboxes = None
   
       # load landmarks
       if landmark_file:
          self.landmarks = np.loadtxt(landmark_file)
       else:
          self.landmarks = None
       
       self.find_three = find_three

    
    def get_basic_item(self, idx):
       img = Image.open(os.path.join(self.img_path, self.img_list[idx]))
       img_id = int(self.img_list[idx].split('/')[3].split('_')[1])

       width, height = img.size
       if self.with_bbox:
          bbox_cor = self.bboxes[idx]
          x1 = max(0, int(bbox_cor[0])-10)
          y1 = max(0, int(bbox_cor[1])-10)
          x2 = int(bbox_cor[2])+10
          y2 = int(bbox_cor[3])+10
          bbox_w = x2-x1
          bbox_h = y2-y1      
          img = img.crop(box=(x1,y1,x2,y2))
       else:
          bbox_w, bbox_h = self.img_size[0], self.img_size[1]

       img.thumbnail(self.img_size, Image.ANTIALIAS)
       img = img.convert('RGB')
       img = self.transform(img)

       label = torch.from_numpy(self.labels[idx])
       landmark = []
       # compute the shifted variety
       origin_landmark = self.landmarks[idx]
       for i, l in enumerate(origin_landmark):
           if i%2==0: # x
              l_x = max(0, l-x1)
              l_x = float(l_x)/width * self.img_size[0]
              landmark.append(l_x) 
           else: # y
              l_y = max(0, l-y1)
              l_y = float(l_y)/height * self.img_size[1]
              landmark.append(l_y)
       landmark = torch.from_numpy(np.array(landmark)).float()
       data = {'img':img,
               'label':label,
               'id':img_id,
               'landmark':landmark}
 
       return data
   

    def get_three_items(self, idx):
        """return anchor, positive and negative 
        """
        anchor_img =  self.img_list[idx]
        anchor_data = self.get_basic_item(idx) 
        anchor_id = int(self.img_list[idx].split('/')[3].split('_')[1])

        # get positive example
        pos_idxes = self.id2idx[anchor_id]
        random_pos_idx = pos_idxes[random.randint(0, len(pos_idxes)-1)]
        while random_pos_idx == idx:
              random_pos_idx = pos_idxes[random.randint(0, len(pos_idxes)-1)]
        pos_data = self.get_basic_item(random_pos_idx)

        # get negative example
        id_len = len(self.ids)
        random_id = self.ids[random.randint(0, id_len-1)]
        while random_id == anchor_id:
              random_id = self.ids[random.randint(0, id_len-1)]
        neg_id = random_id
        neg_idxes = self.id2idx[neg_id]
        neg_idx = random.randint(0, len(neg_idxes)-1)
        neg_data = self.get_basic_item(neg_idx)
  
        data = {'anchor':anchor_data['img'],
                'pos':pos_data['img'],
                'neg':neg_data['img'],
                'anchor_lm':anchor_data['landmark'],
                'pos_lm':pos_data['landmark'],
                'neg_lm':neg_data['landmark']}
        return data 
         
    def __getitem__(self, idx):
        if self.find_three:
           return self.get_three_items(idx)
        else:
           return self.get_basic_item(idx)
   
   
    def __len__(self):
        return len(self.img_list)


