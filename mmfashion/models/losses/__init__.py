from .bce_with_logit_loss import BCEWithLogitsLoss
from .ce_loss import CELoss
from .cosine_embed_loss import CosineEmbeddingLoss
from .l1_loss import L1Loss
from .loss_norm import L1NormLoss, L2NormLoss
from .margin_ranking_loss import MarginRankingLoss, SelectiveMarginLoss
from .mse_loss import MSELoss
from .triplet_loss import TripletLoss
from .vgg_loss import VGGLoss

__all__ = [
    'TripletLoss', 'BCEWithLogitsLoss', 'CELoss', 'CosineEmbeddingLoss',
    'MSELoss', 'MarginRankingLoss', 'SelectiveMarginLoss', 'L1NormLoss',
    'L2NormLoss', 'L1Loss', 'VGGLoss'
]
