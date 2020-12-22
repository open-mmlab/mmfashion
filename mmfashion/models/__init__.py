from .attr_cate_predictor import *  # noqa: F401, F403
from .backbones import *  # noqa: F401, F403
# yapf:disable
from .builder import (build_attr_predictor, build_backbone,
                      build_cate_predictor, build_concat,
                      build_embed_extractor, build_fashion_recommender,
                      build_feature_correlation, build_feature_extractor,
                      build_feature_norm, build_feature_regression,
                      build_geometric_matching, build_global_pool,
                      build_landmark_detector,
                      build_landmark_feature_extractor,
                      build_landmark_regression, build_loss, build_predictor,
                      build_retriever, build_roi_pool, build_tps_warp,
                      build_triplet_net, build_tryon, build_type_specific_net,
                      build_unet_skip_connection_block,
                      build_visibility_classifier)
# yapf:enable
from .concats import *  # noqa: F401, F403
from .embed_extractor import *  # noqa: F401, F403
from .fashion_recommender import *  # noqa: F401, F403
from .feature_extractor import *  # noqa: F401, F403
from .global_pool import *  # noqa: F401, F403
from .landmark_detector import *  # noqa: F401, F403
from .landmark_feature_extractor import *  # noqa: F401, F403
from .landmark_regression import *  # noqa: F401, F403
from .losses import *  # noqa: F401, F403
from .predictor import *  # noqa: F401, F403
from .registry import (ATTRPREDICTOR, BACKBONES, CATEPREDICTOR, CONCATS,
                       EMBEDEXTRACTOR, FEATURECORRELATION, FEATUREEXTRACTOR,
                       FEATURENORM, FEATUREREGRESSION, GEOMETRICMATCHING,
                       GLOBALPOOLING, LANDMARKDETECTOR, LOSSES, PREDICTOR,
                       RECOMMENDER, RETRIEVER, ROIPOOLING, TPSWARP, TRYON,
                       UNETSKIPCONNECTIONBLOCK)
from .retriever import *  # noqa: F401, F403
from .roi_pool import *  # noqa: F401, F403
from .triplet_net import *  # noqa: F401, F403
from .type_specific_net import *  # noqa: F401, F403
from .unet import *  # noqa: F401, F403
from .virtual_tryon import *  # noqa: F401, F403
from .visibility_classifier import *  # noqa: F401, F403

__all__ = [
    'BACKBONES', 'GLOBALPOOLING', 'ROIPOOLING', 'CONCATS', 'LOSSES',
    'PREDICTOR', 'RETRIEVER', 'ATTRPREDICTOR', 'CATEPREDICTOR',
    'EMBEDEXTRACTOR', 'LANDMARKDETECTOR', 'RECOMMENDER', 'FEATUREEXTRACTOR',
    'FEATURECORRELATION', 'FEATUREREGRESSION', 'GEOMETRICMATCHING',
    'FEATURENORM', 'TPSWARP', 'GEOMETRICMATCHING', 'UNETSKIPCONNECTIONBLOCK',
    'TRYON', 'build_backbone', 'build_global_pool', 'build_roi_pool',
    'build_concat', 'build_attr_predictor', 'build_cate_predictor',
    'build_embed_extractor', 'build_predictor', 'build_retriever',
    'build_landmark_feature_extractor', 'build_landmark_regression',
    'build_visibility_classifier', 'build_landmark_detector', 'build_loss',
    'build_triplet_net', 'build_type_specific_net',
    'build_fashion_recommender', 'build_feature_extractor',
    'build_feature_correlation', 'build_feature_norm',
    'build_feature_regression', 'build_tps_warp', 'build_geometric_matching',
    'build_tryon', 'build_unet_skip_connection_block'
]
