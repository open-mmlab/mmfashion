from ..utils import build_from_cfg
from .dataset_wrappers import ConcatDataset
from .registry import DATASETS


def _concat_dataset(cfgs):
    datasets = []
    for cfg in cfgs:
        datasets.append(build_from_cfg(cfg, DATASETS))
    return ConcatDataset(datasets)


def build_dataset(cfg):
    if isinstance(cfg, (list, tuple)):
        dataset = _concat_dataset(cfg)
    else:
        dataset = build_from_cfg(cfg, DATASETS)
    return dataset
