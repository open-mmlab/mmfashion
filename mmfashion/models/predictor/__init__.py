from .base import BasePredictor
from .global_predictor import GlobalPredictor
from .global_attr_cate_predictor import GlobalAttrCatePredictor
from .roi_predictor import RoIPredictor
from .roi_attr_cate_predictor import RoIAttrCatePredictor

__all__ = ['BasePredictor', 'RoIPredictor', 'GlobalPredictor',
           'GlobalAttrCatePredictor', 'RoIAttrCatePredictor']
