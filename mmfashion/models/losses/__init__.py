from .bce_with_logit_loss import BCEWithLogitsLoss
from .ce_loss import CELoss
from .cosine_embed_loss import CosineEmbeddingLoss
from .mse_loss import MSELoss
from .triplet_loss import TripletLoss
from .margin_ranking_loss import MarginRankingLoss, SelectiveMarginLoss
from .loss_norm import L1NormLoss, L2NormLoss

__all__ = [
    'TripletLoss', 'BCEWithLogitsLoss', 'CELoss', 'CosineEmbeddingLoss',
    'MSELoss', 'MarginRankingLoss', 'SelectiveMarginLoss',
    'L1NormLoss', 'L2NormLoss'
]
