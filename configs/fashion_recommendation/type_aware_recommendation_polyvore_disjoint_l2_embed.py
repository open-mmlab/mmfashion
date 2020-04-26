import os

# model settings
arch = 'resnet'
img_size = (224, 224)

model = dict(
    type='TypeAwareRecommender',
    backbone=dict(type='ResNet', setting='resnet18'),
    global_pool=dict(
        type='GlobalPooling',
        inplanes=(7, 7),
        pool_plane=(2, 2),
        inter_channels=[512],
        outchannels=256),
    type_specific_net=dict(
        type='TypeSpecificNet',
        learned=True,
        n_conditions=66,
        rand_typespaces=False,
        use_fc=False,
        l2_embed=True,
        dim_embed=256,
        prein=False),
    triplet_net=dict(
        type='TripletNet',
        text_feature_dim=6000,
        embed_feature_dim=256,
        loss_vse=dict(type='L1NormLoss', loss_weight=5e-3, average=False),
        loss_triplet=dict(
            type='MarginRankingLoss', margin=0.3, loss_weight=5e-4),
        loss_selective_margin=dict(
            type='SelectiveMarginLoss', margin=0.3, loss_weight=5e-4),
        learned_metric=False),
    loss_embed=dict(type='L2NormLoss', loss_weight=5e-4),
    loss_mask=dict(type='L1NormLoss', loss_weight=5e-4),
    pretrained='checkpoint/resnet18.pth')

# dataset setting
dataset_type = 'PolyvoreOutfitDataset'
data_root = 'data/Polyvore'
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    drop_last=False,
    train=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'images'),
        annotation_path=os.path.join(data_root, 'disjoint/train.json'),
        meta_file_path=os.path.join(data_root, 'polyvore_item_metadata.json'),
        img_size=img_size,
        text_feat_path=os.path.join(data_root,
                                    'disjoint/train_hglmm_pca6000.txt'),
        text_feat_dim=6000,
        compatibility_test_fn=os.path.join(data_root,
                                           'disjoint/compatibility_train.txt'),
        fitb_test_fn=os.path.join(data_root,
                                  'disjoint/fill_in_blank_train.json'),
        typespaces_fn=os.path.join(data_root, 'disjoint/typespaces.p'),
        train=True),
    test=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'images'),
        annotation_path=os.path.join(data_root, 'disjoint/test.json'),
        meta_file_path=os.path.join(data_root, 'polyvore_item_metadata.json'),
        img_size=img_size,
        text_feat_path=None,
        text_feat_dim=6000,
        compatibility_test_fn=os.path.join(data_root,
                                           'disjoint/compatibility_test.txt'),
        fitb_test_fn=os.path.join(data_root,
                                  'disjoint/fill_in_blank_test.json'),
        typespaces_fn=os.path.join(data_root, 'disjoint/typespaces.p'),
        train=False),
    val=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'images'),
        annotation_path=os.path.join(data_root, 'disjoint/valid.json'),
        meta_file_path=os.path.join(data_root, 'polyvore_item_metadata.json'),
        img_size=img_size,
        text_feat_path=None,
        text_feat_dim=6000,
        compatibility_test_fn=os.path.join(data_root,
                                           'disjoint/compatibility_valid.txt'),
        fitb_test_fn=os.path.join(data_root,
                                  'disjoint/fill_in_blank_valid.json'),
        typespaces_fn=os.path.join(data_root, 'disjoint/typespaces.p'),
        train=False))

# optimizer
optimizer = dict(type='Adam', lr=5e-5)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[8, 10])

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

start_epoch = 0
total_epochs = 16
gpus = dict(train=[0], test=[0])
work_dir = 'checkpoint/FashionRecommend/TypeAware/disjoint/l2_embed'
print_interval = 20  # interval to print information
save_interval = 5
init_weights_from = 'checkpoint/resnet18.pth'
resume_from = None
load_from = None
workflow = [('train', 10)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
