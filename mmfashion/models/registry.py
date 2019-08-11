from utils import Registry

BACKBONES = Registry('backbone')  # basic feature extractor
GLOBALPOOLING = Registry('global_pool')  # global pooling
ROIPOOLING = Registry('roi_pool')  # roi pooling
CONCATS = Registry('concat')  # concat local features and global features
ATTRPREDICTOR = Registry('attr_predictor') # predict attributes
EMBEDEXTRACTOR = Registry('embed_extractor') # extract embeddings

LOSSES = Registry('loss')  # loss function

PREDICTOR = Registry('predictor')

RETRIEVER = Registry('retriever')
