from __future__ import division
import os

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

from .registry import DATASETS


@DATASETS.register_module
class AttrDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 img_path,
                 img_file,
                 label_file,
                 cate_file,
                 bbox_file,
                 landmark_file,
                 img_size,
                 idx2id=None):
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

        # read attribute labels and category annotations
        self.labels = np.loadtxt(label_file, dtype=np.float32)

        # read categories
        self.categories = []
        catefn = open(cate_file).readlines()
        for i, line in enumerate(catefn):
            self.categories.append(line.strip('\n'))

        self.img_size = img_size

        # load bbox
        if bbox_file:
            self.with_bbox = True
            self.bboxes = np.loadtxt(bbox_file, usecols=(0, 1, 2, 3))
        else:
            self.with_bbox = False
            self.bboxes = None

        # load landmarks
        if landmark_file:
            self.landmarks = np.loadtxt(landmark_file)
        else:
            self.landmarks = None

    def get_basic_item(self, idx):
        img = Image.open(os.path.join(self.img_path,
                                      self.img_list[idx])).convert('RGB')

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

        label = torch.from_numpy(self.labels[idx])
        cate = torch.LongTensor([int(self.categories[idx]) - 1])

        landmark = []
        # compute the shiftness
        if self.landmarks is not None:
            origin_landmark = self.landmarks[idx]
            for i, l in enumerate(origin_landmark):
                if i % 2 == 0:  # x
                    l_x = max(0, l - x1)
                    l_x = float(l_x) / bbox_w * self.img_size[0]
                    landmark.append(l_x)
                else:  # y
                    l_y = max(0, l - y1)
                    l_y = float(l_y) / bbox_h * self.img_size[1]
                    landmark.append(l_y)
            landmark = torch.from_numpy(np.array(landmark)).float()
        else:
            # here no landmark will be used, just use zero for initialization
            # (global predictor)
            landmark = torch.zeros(8)
        
        data = {'img': img, 'attr': label, 'cate': cate, 'landmark': landmark}
        return data

    def __getitem__(self, idx):
        return self.get_basic_item(idx)

    def __len__(self):
        return len(self.img_list)
