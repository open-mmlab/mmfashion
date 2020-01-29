from .base import BasePredictor
from .. import builder
from ..registry import PREDICTOR


@PREDICTOR.register_module
class GlobalPredictor(BasePredictor):

    def __init__(self,
                 backbone,
                 global_pool,
                 attr_predictor,
                 loss_attr=dict(
                     type='BCEWithLogitsLoss',
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean'),
                 roi_pool=None,
                 pretrained=None):
        super(GlobalPredictor, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
        self.attr_predictor = builder.build_attr_predictor(attr_predictor)

        self.loss_attr = builder.build_loss(loss_attr)

    def forward_train(self, x, landmarks, attr):
        # 1. conv layers extract global features
        x = self.backbone(x)

        # 2. global pooling
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)

        attr_pred = self.attr_predictor(global_x)
        losses = dict()
        losses['loss_attr'] = self.loss_attr(attr_pred, attr)

        return losses

    def simple_test(self, x, landmarks):
        """Test single image"""
        x = x.unsqueeze(0)
        attr_pred = self.aug_test(x)[0]
        return attr_pred

    def aug_test(self, x, landmarks):
        """Test batch of images"""
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        attr_pred = self.attr_predictor(global_x)
        return attr_pred

    def forward_test(self, x, landmarks):
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        attr_pred = self.attr_predictor(global_x)
        return attr_pred

    def init_weights(self, pretrained=None):
        super(GlobalPredictor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights()
        self.attr_predictor.init_weights()
