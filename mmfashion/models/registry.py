from ..utils import Registry

BACKBONES = Registry('backbone')
GLOBALPOOLING = Registry('global_pool')  # global pooling
ROIPOOLING = Registry('roi_pool')  # roi pooling
CONCATS = Registry('concat')  # concat local features and global features
ATTRPREDICTOR = Registry('attr_predictor')  # predict attributes
CATEPREDICTOR = Registry('cate_predictor') # predict category
EMBEDEXTRACTOR = Registry('embed_extractor')  # extract embeddings

LANDMARKFEATUREEXTRACTOR = Registry('landmark_feature_extractor')
VISIBILITYCLASSIFIER = Registry('visibility_classifier')
LANDMARKREGRESSION = Registry('landmark_regression')

LOSSES = Registry('loss')  # loss function

PREDICTOR = Registry('predictor')

RETRIEVER = Registry('retriever')

LANDMARKDETECTOR = Registry('landmark_detector')

TYPESPECIFICNET = Registry('type_specific_net')
TRIPLETNET = Registry('triplet_net')

RECOMMENDER = Registry('fashion_recommender')

FEATUREEXTRACTOR = Registry('feature_extractor')
FEATURENORM = Registry('feature_norm')
FEATURECORRELATION = Registry('feature_correlation')
FEATUREREGRESSION = Registry('feature_regression')

TPSWARP = Registry('tps_warp')

GEOMETRICMATCHING = Registry('geometric_matching')