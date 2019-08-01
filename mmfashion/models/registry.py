from ..utils import Registry

BACKBONES = Registry('backbone')  # basic feature extractor
GLOBALPOOLING = Registry('global_pool')  # global pooling
ROIPOOLING = Registry('roi_pool')  # roi pooling
CONCATS = Registry('concat')  # concat local features and global features
LOSSES = Registry('loss')  # loss function

PREDICTOR = Registry('predictor')

RETRIEVER = Registry('retriever')
