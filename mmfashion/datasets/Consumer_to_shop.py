import os
import random

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

from .registry import DATASETS


@DATASETS.register_module
class ConsumerToShopDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 img_path,
                 img_file,
                 id_file,
                 label_file,
                 bbox_file,
                 landmark_file,
                 img_size,
                 roi_plane_size=7,
                 retrieve=False,
                 find_three=False):
        self.img_path = img_path

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # collect the list of id label(int)
        self.ids = []
        id_fn = open(id_file).readlines()
        for idx, line in enumerate(id_fn):
            img_id = int(line.strip('\n'))
            self.ids.append(img_id)

        if find_three:  # during train
            # read train set: image pairs(consumer.jpg, shop.jpg)
            img_pairs = open(img_file).readlines()
            self.consumer2shop = {}
            self.img_list = []
            self.id_list = []  # collect the list of 'id_xxxxx'(str)
            self.consumer_id2img, self.shop_id2img = {}, {}
            for i, line in enumerate(img_pairs):
                aline = line.strip('\n').split()
                consumer, shop = aline[0], aline[1]
                self.img_list.append(consumer)  # consumer
                self.consumer2shop[consumer] = shop  # shop
                consumer_id = consumer.split(
                    '/'
                )[3]  # e.g. 'id_00006282', not exactly same with self.ids[idx]
                shop_id = shop.split('/')[3]
                self.id_list.append(consumer_id)

                if consumer_id not in self.consumer_id2img:
                    self.consumer_id2img[consumer_id] = [consumer]
                else:
                    self.consumer_id2img[consumer_id].append(consumer)

                if shop_id not in self.shop_id2img:
                    self.shop_id2img[shop_id] = [shop]
                else:
                    self.shop_id2img[shop_id].append(shop)

        else:  # during test
            img_lines = open(img_file).readlines()
            self.img_list = []
            self.id2img = {}
            for i, line in enumerate(img_lines):
                img = line.strip('\n')
                self.img_list.append(img)
                one_id = img.split('/')[3]
                if one_id in self.id2img:
                    self.id2img[one_id].append(img)
                else:
                    self.id2img[one_id] = [img]

        # read attribute file
        if label_file is not None:
            self.with_label = True
            attribute_list = open(label_file).readlines()[2:]
            self.id2attribute = {}
            for i, line in enumerate(attribute_list):
                aline = line.strip('\n').split()
                one_id = aline[0]
                attributes = aline[1:]
                one_attr = []
                for a in attributes:
                    if a == 1:
                        one_attr.append(1)
                    else:
                        one_attr.append(0)
                self.id2attribute[one_id] = np.asarray(one_attr)
        else:
            self.with_label = False

        # read bbox file
        if bbox_file is not None:
            self.with_bbox = True
            bbox_list = open(bbox_file).readlines()
            self.img2bbox = {}
            for i, line in enumerate(bbox_list):
                aline = line.strip('\n').split()
                img = aline[0]
                bbox = []
                for bi, b in enumerate(aline[1:]):
                    bbox.append(int(b))
                self.img2bbox[img] = np.asarray(bbox)
        else:
            self.with_bbox = False

        # read landmarks
        if landmark_file is not None:
            self.with_landmark = True
            landmark_list = open(landmark_file).readlines()
            self.img2landmark = {}
            for i, line in enumerate(landmark_list):
                aline = line.strip('\n').split()
                img = aline[0]
                landmark = []
                for li, l in enumerate(aline[1:]):
                    landmark.append(int(l))
                self.img2landmark[img] = np.asarray(landmark)
        else:
            self.with_landmark = False

        self.roi_plane_size = roi_plane_size
        self.img_size = img_size
        self.find_three = find_three  # true for train, false for test

    def get_basic_item(self, img_name):
        ''' get one consumer / shop item'''
        img = Image.open(os.path.join(self.img_path, img_name))
        width, height = img.size
        if self.with_bbox:
            bbox_cor = self.img2bbox[img_name]
            x1 = max(0, bbox_cor[0] - 20)
            y1 = max(0, bbox_cor[1] - 20)
            x2 = min(bbox_cor[2] + 20, width)
            y2 = min(bbox_cor[3] + 20, height)
            img = img.crop(box=(x1, y1, x2, y2))

        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = img.convert('RGB')

        one_id = img_name.split('/')[3]
        attributes = torch.from_numpy(self.id2attribute[one_id])

        origin_landmark = self.img2landmark[img_name]
        landmark = []
        # compute the shifted variety
        for i, l in enumerate(origin_landmark):
            if i % 2 == 0:  # x
                l_x = max(0, l - x1)
                l_x = float(l_x) / width * self.roi_plane_size
                landmark.append(l_x)
            else:  # y
                l_y = max(0, l - y1)
                l_y = float(l_y) / height * self.roi_plane_size
                landmark.append(l_y)

        landmark = torch.from_numpy(np.asarray(landmark, dtype=np.float32))

        img = self.transform(img)

        data = {'img': img, 'landmark': landmark, 'attr': attributes}
        return data

    def get_three_items(self, idx):
        ''' get consumer item, positive shop item and negative shop item'''
        consumer = self.img_list[idx]
        consumer_data = self.get_basic_item(consumer)

        # get positive example
        shop = self.consumer2shop[consumer]
        shop_data = self.get_basic_item(shop)

        # get negative example
        id_len = len(self.id_list)
        consumer_id = self.id_list[idx]
        random_id = self.id_list[random.randint(0, id_len - 1)]
        while random_id == consumer_id:
            random_id = self.id_list[random.randint(0, id_len - 1)]
        neg_id = random_id
        neg_id_len = len(self.shop_id2img[neg_id])
        neg_shop = self.shop_id2img[neg_id][random.randint(0, neg_id_len - 1)]
        neg_shop_data = self.get_basic_item(neg_shop)

        # create label for triplet loss
        triplet_pos_label = np.float32(1)
        triplet_neg_label = np.float32(-1)

        # get id
        consumer_id_label = self.ids[idx]

        data = {
            'img': consumer_data['img'],
            'landmark': consumer_data['landmark'],
            'id': consumer_id_label,
            'attr': consumer_data['attr'],
            'pos': shop_data['img'],
            'neg': neg_shop_data['img'],
            'pos_lm': shop_data['landmark'],
            'neg_lm': neg_shop_data['landmark'],
            'triplet_pos_label': triplet_pos_label,
            'triplet_neg_label': triplet_neg_label
        }
        return data

    def __getitem__(self, idx):
        if self.find_three:
            return self.get_three_items(idx)
        else:
            id_label = self.ids[idx]
            basic_data = self.get_basic_item(self.img_list[idx])
            return {
                'img': basic_data['img'],
                'landmark': basic_data['landmark'],
                'id': id_label,
                'attr': basic_data['attr']
            }

    def __len__(self):
        return len(self.img_list)
