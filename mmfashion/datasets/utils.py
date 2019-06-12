from __future__ import division
import shutil
import time
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

from .BasicDataset import BasicDataset
from .ThreedDataset import ThreedDataset

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def get_data(cfg, traindata):
    imgs, target = Variable(traindata[0]).cuda(), Variable(traindata[1]).cuda()
    if cfg.pooling == 'RoI':
       landmarks = Variable(traindata[2]).cuda()
       iuv = None
    elif cfg.pooling =='IUV':
       landmarks = Variable(traindata[2]).cuda()
       iuv = traindata[3]
       if iuv is not None:
          iuv = Variable(iuv).cuda()
       else:
          iuv = None
    return imgs, target, landmarks, iuv



def get_dataset(data_cfg):
    if data_cfg['type'] == 'roi_dataset':
       dataset = BasicDataset(data_cfg.img_path, data_cfg.img_file,
                              data_cfg.label_file, data_cfg.bbox_file,
                              data_cfg.landmark_file, data_cfg.img_scale)

    elif data_cfg['type'] == 'iuv_dataset':
       dataset = ThreedDataset(data_cfg.img_path, data_cfg.img_file,
                               data_cfg.label_file, data_cfg.bbox_file,
                               data_cfg.landmark_file, data_cfg.iuv_file,
                               data_cfg.img_scale)      
    else:
       raise TypeError('type {} does not exist.'.fomart(data_cfg['type']))

    return dataset 
