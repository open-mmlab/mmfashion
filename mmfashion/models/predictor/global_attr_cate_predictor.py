from .. import builder
from ..registry import PREDICTOR
from .base import BasePredictor


@PREDICTOR.register_module
class GlobalAttrCatePredictor(BasePredictor):

    def __init__(self,
                 backbone,
                 global_pool,
                 attr_predictor,
                 cate_predictor,
                 roi_pool=None,
                 pretrained=None):
        super(GlobalAttrCatePredictor, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
        self.attr_predictor = builder.build_attr_predictor(attr_predictor)
        self.cate_predictor = builder.build_cate_predictor(cate_predictor)

        self.init_weights(pretrained)
        
    def forward_train(self, x, landmarks, attr, cate):
        # landmarks will not be used in global predictor
        # 1. conv layers extract global features
        x = self.backbone(x)

        # 2. global pooling
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)

        loss_attr = self.attr_predictor(global_x, attr, return_loss=True)
        loss_cate = self.cate_predictor(global_x, cate, return_loss=True)
        losses = dict()
        losses['loss_attr'] = loss_attr
        losses['loss_cate'] = loss_cate
        return losses

    def simple_test(self, x, landmarks):
        """Test single image"""
        x = x.unsqueeze(0)
        attr_pred, cate_pred = self.aug_test(x)[0]
        return attr_pred, cate_pred

    def aug_test(self, x, landmarks):
        """Test batch of images"""
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        attr_pred = self.attr_predictor(global_x)
        cate_pred = self.cate_predictor(global_x)
        return attr_pred, cate_pred

    def forward_test(self, x, landmarks):
        # landmarks will not be used
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        attr_pred = self.attr_predictor(global_x)
        cate_pred = self.cate_predictor(global_x)
        return attr_pred, cate_pred

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights()
        self.attr_predictor.init_weights()
        self.cate_predictor.init_weights()
