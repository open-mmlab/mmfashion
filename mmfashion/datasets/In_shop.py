import os
import random

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
class InShopDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 img_path,
                 img_file,
                 label_file,
                 id_file,
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

        # read img names
        fp = open(img_file, 'r')
        self.img_list = [x.strip() for x in fp]

        # collect id
        self.ids = []
        id_fn = open(id_file).readlines()
        self.id2idx, self.idx2id = {}, {}
        for idx, line in enumerate(id_fn):
            img_id = int(line.strip('\n'))
            self.ids.append(img_id)
            self.idx2id[idx] = img_id

            if img_id not in self.id2idx:
                self.id2idx[img_id] = [idx]
            else:
                self.id2idx[img_id].append(idx)
        fp.close()

        # read labels
        self.labels = np.loadtxt(label_file, dtype=np.float32)

        self.img_size = img_size
        self.roi_plane_size = roi_plane_size

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

        self.find_three = find_three

    def get_basic_item(self, idx):
        img = Image.open(os.path.join(self.img_path, self.img_list[idx]))
        img_id = self.ids[idx]
        width, height = img.size

        if self.with_bbox:
            bbox_cor = self.bboxes[idx]
            x1 = max(0, int(bbox_cor[0]) - 20)
            y1 = max(0, int(bbox_cor[1]) - 20)
            x2 = int(bbox_cor[2]) + 20
            y2 = int(bbox_cor[3]) + 20
            img = img.crop(box=(x1, y1, x2, y2))

        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = img.convert('RGB')

        label = torch.from_numpy(self.labels[idx])
        landmark = []
        # compute the shifted variety

        origin_landmark = self.landmarks[idx]
        for i, l in enumerate(origin_landmark):
            if i % 2 == 0:  # x
                l_x = max(0, l - x1)
                l_x = float(l_x) / width * self.roi_plane_size
                landmark.append(l_x)
            else:  # y
                l_y = max(0, l - y1)
                l_y = float(l_y) / height * self.roi_plane_size
                landmark.append(l_y)

        landmark = torch.from_numpy(np.array(landmark)).float()
        img = self.transform(img)
        data = {'img': img, 'landmark': landmark, 'id': img_id, 'attr': label}
        return data

    def get_three_items(self, idx):
        """return anchor, positive and negative
        """
        anchor_data = self.get_basic_item(idx)
        anchor_id = self.ids[idx]

        # get positive example
        pos_idxes = self.id2idx[anchor_id]
        if len(pos_idxes) == 1:  # just one item
            pos_data = anchor_data
        else:
            random_pos_idx = pos_idxes[random.randint(0, len(pos_idxes) - 1)]
            while random_pos_idx == idx:
                random_pos_idx = pos_idxes[random.randint(
                    0,
                    len(pos_idxes) - 1)]
            pos_data = self.get_basic_item(random_pos_idx)

        # get negative example
        id_len = len(self.ids)
        random_id = self.ids[random.randint(0, id_len - 1)]
        while random_id == anchor_id:
            random_id = self.ids[random.randint(0, id_len - 1)]
        neg_id = random_id
        neg_idxes = self.id2idx[neg_id]
        neg_idx = neg_idxes[random.randint(0, len(neg_idxes) - 1)]
        neg_data = self.get_basic_item(neg_idx)

        # create label for triplet loss
        triplet_pos_label = np.float32(1)
        triplet_neg_label = np.float32(-1)

        data = {
            'img': anchor_data['img'],
            'landmark': anchor_data['landmark'],
            'id': anchor_data['id'],
            'attr': anchor_data['attr'],
            'pos': pos_data['img'],
            'neg': neg_data['img'],
            'pos_lm': pos_data['landmark'],
            'neg_lm': neg_data['landmark'],
            'triplet_pos_label': triplet_pos_label,
            'triplet_neg_label': triplet_neg_label
        }
        return data

    def __getitem__(self, idx):
        if self.find_three:
            return self.get_three_items(idx)
        else:
            return self.get_basic_item(idx)

    def __len__(self):
        return len(self.img_list)
