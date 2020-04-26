from __future__ import division
from collections.abc import Sequence

import mmcv
import numpy as np
import torch

from .Attr_Pred import AttrDataset
from .Consumer_to_shop import ConsumerToShopDataset
from .In_shop import InShopDataset
from .Landmark_Detect import LandmarkDetectDataset
from .Polyvore_outfit import PolyvoreOutfitDataset


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
                                data_cfg.label_file, data_cfg.id_file,
                                data_cfg.bbox_file, data_cfg.landmark_file,
                                data_cfg.img_size, data_cfg.roi_plane_size,
                                data_cfg.find_three)

    elif data_cfg['type'] == 'Consumer_to_shop':
        dataset = ConsumerToShopDataset(
            data_cfg.img_path, data_cfg.img_file, data_cfg.label_file,
            data_cfg.bbox_file, data_cfg.landmark_file, data_cfg.img_size,
            data_cfg.roi_plane_size, data_cfg.find_three)

    elif data_cfg['type'] == 'Attr_Pred':
        dataset = AttrDataset(data_cfg.img_path, data_cfg.img_file,
                              data_cfg.label_file, data_cfg.cate_file,
                              data_cfg.bbox_file, data_cfg.landmark_file,
                              data_cfg.img_size)

    elif data_cfg['type'] == 'Landmark_Detect':
        dataset = LandmarkDetectDataset(data_cfg.img_path, data_cfg.img_file,
                                        data_cfg.bbox_file,
                                        data_cfg.landmark_file,
                                        data_cfg.img_size)
    elif data_cfg['type'] == 'PolyvoreOutfitDataset':
        dataset = PolyvoreOutfitDataset(
            data_cfg.img_path, data_cfg.annotation_path,
            data_cfg.meta_file_path, data_cfg.img_size,
            data_cfg.text_feat_path, data_cfg.text_feat_dim,
            data_cfg.compatibility_test_fn, data_cfg.fitb_test_fn,
            data_cfg.typespaces_fn, data_cfg.train)
    else:
        raise TypeError('type {} does not exist.'.format(data_cfg['type']))

    return dataset
