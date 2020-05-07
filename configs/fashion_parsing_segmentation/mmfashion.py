from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class mmfashionDataset(CocoDataset):
    CLASSES = ('top', 'skirt', 'leggings', 'dress', 'outer', 'pants', 'bag',
               'neckwear', 'headwear', 'eyeglass', 'belt', 'footwear', 'hair',
               'skin', 'face')
