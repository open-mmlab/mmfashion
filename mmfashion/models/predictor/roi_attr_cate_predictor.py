from .. import builder
from ..registry import PREDICTOR
from .base import BasePredictor


@PREDICTOR.register_module
class RoIAttrCatePredictor(BasePredictor):

    def __init__(self,
                 backbone,
                 global_pool,
                 roi_pool,
                 concat,
                 attr_predictor,
                 cate_predictor,
                 pretrained=None):
        super(RoIAttrCatePredictor, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)

        assert roi_pool is not None
        self.roi_pool = builder.build_roi_pool(roi_pool)

        self.concat = builder.build_concat(concat)
        self.attr_predictor = builder.build_attr_predictor(attr_predictor)
        self.cate_predictor = builder.build_cate_predictor(cate_predictor)

        self.init_weights(pretrained)

    def forward_train(self, x, landmark, attr, cate):
        # 1. conv layers extract global features
        x = self.backbone(x)

        # 2. global pooling
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)

        # 3. roi pooling
        local_x = self.roi_pool(x, landmark)

        # 4. concat
        feat = self.concat(global_x, local_x)

        # 5. attribute prediction
        losses = dict()
        losses['loss_attr'] = self.attr_predictor(feat, attr, return_loss=True)
        losses['loss_cate'] = self.cate_predictor(feat, cate, return_loss=True)

        return losses

    def simple_test(self, x, landmark=None):
        """Test single image"""
        x = x.unsqueeze(0)
        landmark = landmark.unsqueeze(0)
        attr_pred, cate_pred = self.aug_test(x, landmark)
        return attr_pred[0], cate_pred[0]

    def aug_test(self, x, landmark=None):
        """Test batch of images"""
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        local_x = self.roi_pool(x, landmark)

        feat = self.concat(global_x, local_x)
        attr_pred = self.attr_predictor(feat)
        cate_pred = self.cate_predictor(feat)
        return attr_pred, cate_pred

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights()
        self.roi_pool.init_weights()
        self.concat.init_weights()
        self.attr_predictor.init_weights()
        self.cate_predictor.init_weights()
