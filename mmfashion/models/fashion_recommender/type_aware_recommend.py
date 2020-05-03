from .. import builder
from ..registry import RECOMMENDER
from .base import BaseFashionRecommender


@RECOMMENDER.register_module
class TypeAwareRecommender(BaseFashionRecommender):

    def __init__(self,
                 backbone,
                 global_pool,
                 type_specific_net,
                 triplet_net,
                 loss_embed=dict(type='L2NormLoss', loss_weight=5e-4),
                 loss_mask=dict(type='L1NormLoss', loss_weight=5e-4),
                 pretrained=None):
        super(TypeAwareRecommender, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
        self.type_specific_net = builder.build_type_specific_net(
            type_specific_net)
        self.triplet_net = builder.build_triplet_net(triplet_net)

        self.loss_embed = builder.build_loss(loss_embed)
        self.loss_mask = builder.build_loss(loss_mask)

        self.init_weights(pretrained=pretrained)

    def forward_train(self, img, text, has_text, pos_img, pos_text,
                      pos_has_text, neg_img, neg_text, neg_has_text,
                      condition):

        embed_x = self.backbone(img)
        embed_z = self.backbone(pos_img)  # close
        embed_y = self.backbone(neg_img)  # far
        embed_x = self.global_pool(embed_x)
        embed_y = self.global_pool(embed_y)
        embed_z = self.global_pool(embed_z)

        masked_embed_x, mask_norm_x, embed_norm_x = self.type_specific_net(
            embed_x, condition, return_loss=True)
        masked_embed_y, mask_norm_y, embed_norm_y = self.type_specific_net(
            embed_y, condition, return_loss=True)
        masked_embed_z, mask_norm_z, embed_norm_z = self.type_specific_net(
            embed_z, condition, return_loss=True)

        loss_embed = self.loss_embed(embed_norm_x, embed_norm_y, embed_norm_z,
                                     len(img))
        loss_mask = self.loss_mask(mask_norm_x, mask_norm_y, mask_norm_z,
                                   len(img))

        loss_triplet, loss_sim_t, loss_vse, loss_sim_i = self.triplet_net(
            embed_x, masked_embed_x, text, has_text, embed_y, masked_embed_y,
            neg_text, neg_has_text, embed_z, masked_embed_z, pos_text,
            pos_has_text)

        loss_sim = loss_sim_i + loss_sim_t
        loss_reg = loss_embed + loss_mask
        losses = {
            'loss_triplet': loss_triplet,
            'loss_sim': loss_sim,
            'loss_reg': loss_reg,
            'loss_vse': loss_vse
        }
        return losses

    def forward_test(self, img):
        embed = self.backbone(img)
        embed = self.global_pool(embed)
        type_aware_embed = self.type_specific_net(embed, return_loss=False)
        return type_aware_embed

    def init_weights(self, pretrained=None):
        super(TypeAwareRecommender, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights()
        self.type_specific_net.init_weights()
        self.triplet_net.init_weights()
