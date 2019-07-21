import copy

from utils import build_from_cfg
from .registry import DATASETS

def build_dataset(cfg):
    return build_from_cfg(cfg, DATASETS)
