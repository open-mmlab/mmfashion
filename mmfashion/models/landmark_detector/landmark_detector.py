from .. import builder
from ..registry import LANDMARKDETECTOR
from .base import BaseLandmarkDetector


@LANDMARKDETECTOR.register_module
class LandmarkDetector(BaseLandmarkDetector):

    def __init__(self,
                 backbone,
                 global_pool,
                 landmark_feature_extractor,
                 visibility_classifier,
                 landmark_regression,
                 pretrained=None):

        super(LandmarkDetector, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
        self.landmark_feature_extractor = \
            builder.build_landmark_feature_extractor(
                landmark_feature_extractor)
        self.visibility_classifier = builder.build_visibility_classifier(
            visibility_classifier)
        self.landmark_regression = builder.build_landmark_regression(
            landmark_regression)

        self.init_weights(pretrained=pretrained)

    def forward_train(self,
                      x,
                      vis,
                      landmark_for_regression,
                      landmark_for_roi_pool=None,
                      attr=None):
        x = self.backbone(x)
        x = self.global_pool(x)

        # extract landmark features
        landmark_feat = self.landmark_feature_extractor(x)

        losses = dict()
        # predict landmark visibility
        losses['loss_vis'], pred_vis = self.visibility_classifier(
            landmark_feat, vis)
        # predict landmark coordinates
        losses['loss_regress'] = self.landmark_regression(
            landmark_feat, pred_vis, vis, landmark_for_regression)
        return losses

    def simple_test(self, x):
        x = x.unsqueeze(0)
        x = self.backbone(x)
        x = self.global_pool(x)
        landmark_feat = self.landmark_feature_extractor(x)
        pred_vis = self.visibility_classifier(landmark_feat, return_loss=False)
        pred_lm = self.landmark_regression(
            landmark_feat, pred_vis, return_loss=False)
        return pred_vis[0], pred_lm[0]

    def aug_test(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        landmark_feat = self.landmark_feature_extractor(x)
        pred_vis = self.visibility_classifier(landmark_feat, return_loss=False)
        pred_lm = self.landmark_regression(
            landmark_feat, pred_vis, return_loss=False)
        return pred_vis, pred_lm

    def init_weights(self, pretrained=None):
        super(LandmarkDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights()
        self.landmark_feature_extractor.init_weights()
        self.visibility_classifier.init_weights()
        self.landmark_regression.init_weights()
