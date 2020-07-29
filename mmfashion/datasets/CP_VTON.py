from __future__ import division
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from PIL import Image, ImageDraw
import numpy as np
import json

from .registry import DATASETS

@DATASETS.register_module
class CPVTONDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 dataroot,
                 datamode,
                 stage,
                 data_list,
                 fine_height=256,
                 fine_width=192,
                 radius=5):
        super(CPVTONDataset, self).__init__()
        self.dataroot = dataroot
        self.datamode = datamode # train, test, self-defined
        self.stage = stage
        self.data_list = data_list
        self.fine_height = fine_height
        self.fine_width = fine_width
        self.radius = radius
        self.data_path = os.path.join(self.dataroot, datamode)

        normalize = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # load data list
        im_names = []
        c_names = []
        with open(os.path.join(self.dataroot, data_list), 'r') as rf:
            for line in rf.readlines():
                # get person img and in-shop cloth c
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        self.im_names = im_names
        self.c_names = c_names

    def __getitem__(self, index):
        # get data for GMM stage
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        if self.stage == 'GMM':
            c = Image.open(os.path.join(self.data_path, 'cloth', c_name))
            cm = Image.open(os.path.join(self.data_path, 'cloth-mask', c_name))
        else:
            c = Image.open(os.path.join(self.data_path, 'warp-cloth', c_name))
            cm = Image.open(os.path.join(self.data_path, 'warp-mask', c_name))
        c = self.transform(c)
        cm_array = np.array(cm)
        cm_array = (cm_array>128).astype(np.float32)
        cm = torch.from_numpy(cm_array).unsqueeze_(0)

        # person image
        im = Image.open(os.path.join(self.data_path, 'image', im_name))
        im = self.transform(im)

        # parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(os.path.join(self.data_path, 'image-parse', parse_name))
        parse_array = np.array(im_parse)
        parse_shape = (parse_array>0).astype(np.float32)
        parse_head = (parse_array==1).astype(np.float32) \
                     + (parse_array==2).astype(np.float32) \
                     + (parse_array==4).astype(np.float32) \
                     + (parse_array==13).astype(np.float32)
        parse_cloth = (parse_array==5).astype(np.float32) \
                     + (parse_array==6).astype(np.float32) \
                     + (parse_array==7).astype(np.float32)

        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform(parse_shape)
        phead = torch.from_numpy(parse_head)
        pcm = torch.from_numpy(parse_cloth)

        # upper cloth
        im_c = im*pcm + (1-pcm) # [-1,1], fill 1 for other parts
        im_h = im*phead - (1-phead) # [-1,1], fill 0 for other parts

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(self.data_path, 'pose', pose_name), 'r') as rf:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # for visualization
        im_pose = self.transform(im_pose)

        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0)

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''

        data = {
            'c_name': c_name,
            'im_name': im_name,
            'cloth': c,
            'cloth_mask': cm,
            'image': im,
            'agnostic': agnostic,
            'parse_cloth': im_c,
            'shape': shape,
            'head': im_h,
            'pose_image': im_pose,
            'grid_image': im_g
        }
        return data

    def __len__(self):
        return len(self.im_names)