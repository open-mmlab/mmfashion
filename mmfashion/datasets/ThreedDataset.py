import torch
import os
import sys
from skimage import io
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import cv2
from pdb import set_trace as bp


class ThreedDataset(Dataset):
    def __init__(self, img_path, img_file, label_file, bbox_file, landmark_file, iuv_file, img_size):
        
        self.img_path = img_path
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize,
        ])
  
        # read img names
        fp = open(img_file, 'r')
        self.img_list = [x.strip() for x in fp]
        fp.close()

        # read labels
        self.labels = np.loadtxt(label_file, dtype=np.float32)

        self.img_size = img_size

        # load bbox
        if bbox_file:
           self.with_bbox = True
           self.bboxes = np.loadtxt(bbox_fn, usecols=(0,1,2,3))
        else:
           self.with_bbox = False
           self.bboxes = None
        
        if landmark_file:
           self.landmarks = np.loadtxt(landmark_file)
        else:
           self.landmarks = None

        iuv_fp = open(iuv_file, 'r')
        self.iuv_list = [x.strip() for x in iuv_fp]
        iuv_fp.close()

    def __getitem__(self,idx):
        # read image
        img = Image.open(os.path.join(self.img_path, self.img_list[idx])).convert('RGB')
        width, height = img.size

        # crop the bbox region
        if self.with_bbox:
            bbox_cor = self.bboxes[idx]
            x1 = max(0,int(bbox_cor[0])-10)
            y1 = max(0,int(bbox_cor[1])-10)
            x2 = min(int(bbox_cor[2])+10, cfg.width)
            y2 = min(int(bbox_cor[3])+10, cfg.height)
            bbox_w, bbox_h = x2-x1, y2-y1
            img = img.crop(box=(x1,y1,x2,y2))
        else:
            bbox_w, bbox_h = cfg.width, cfg.height

        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = self.transform(img)
        label = torch.from_numpy(self.labels[idx])
  
        # read iuv corordinates
        # [width, height, channel]
        # iuv_img[:,:,0] = partidx; iuv_img[:,:,1] = x; iuv_img[:,:,2] = y
        iuv = np.zeros((22,2)) # 22 parts, x & y
        iuv_img = cv2.imread(os.path.join(self.img_path, self.iuv_list[idx]))
        if iuv_img is not None:
           for idx in range(1,23): # except face
               c = np.where(iuv_img[:,:,0] == idx)
               x_arr, y_arr = c[0], c[1]
               if len(x_arr)>0 and len(y_arr)>0:
                  x, y = np.average(x_arr), np.average(y_arr) 
                  x =  float(max(0, x-x1))/width* cfg.width
                  y = float(max(0, y-y1))/height* cfg.height
                  iuv[idx-1][0] = x
                  iuv[idx-1][1] = y
        iuv = torch.from_numpy(iuv)
        return img, label, iuv

    def __len__(self):
        return len(self.img_list)


#train_loader = ThreedDataLoader(batch_size=16, num_workers=4).train_loader
#print('training data loaded')
#test_loader = ThreedDataLoader(batch_size=16, num_workers=4).test_loader
#print('testing data loaded')
