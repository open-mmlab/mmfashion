from __future__ import division
import shutil
import time
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

from .In_shop import InShopDataset
from .Attr_Pred import AttrDataset


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


def get_dataset(data_cfg):
    if data_cfg['type'] == 'In-shop':
        dataset = InShopDataset(data_cfg.img_path, data_cfg.img_file,
                                data_cfg.label_file, 
                                data_cfg.id_file,
                                data_cfg.bbox_file,
                                data_cfg.landmark_file, data_cfg.img_size,
                                data_cfg.roi_plane_size,
                                data_cfg.find_three)
    elif data_cfg['type'] == 'Attr_Pred':
        dataset = AttrDataset(data_cfg.img_path, data_cfg.img_file,
                              data_cfg.label_file,
                              data_cfg.cate_file, data_cfg.bbox_file,
                              data_cfg.landmark_file, data_cfg.img_size)
    else:
        raise TypeError('type {} does not exist.'.format(data_cfg['type']))

    return dataset
