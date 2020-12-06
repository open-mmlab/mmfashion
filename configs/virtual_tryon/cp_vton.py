import os
import torch.nn as nn

# model setting
# Geometric Matching Module
GMM = dict(
    type='GeometricMatching',
    feature_extractor_a=dict(
        type='FeatureExtractor',
        in_channels=22,
        ngf=64,
        n_layers=3),
    feature_extractor_b=dict(
        type='FeatureExtractor',
        in_channels=3,
        ngf=64,
        n_layers=3),
    feature_norm=dict(
        type='FeatureNorm',
        eps=1e-6),
    feature_correlation=dict(
        type='FeatureCorrelation'),
    feature_regression=dict(
        type='FeatureRegression',
        in_channels=192,
        out_channels=18,
        inter_channels=(512, 256, 128, 64)),
    tps_warp=dict(
        type='TPSWarp',
        out_h=256,
        out_w=192,
        use_regular_grid=True,
        grid_size=3,
        reg_factor=0),
    loss=dict(type='L1Loss'),
    pretrained=None)

# TryON Module
TOM = dict(
    type='Tryon',
    ngf=64,
    num_downs=6,
    in_channels=25,
    out_channels=4,
    down_channels=(8, 8),
    inter_channels=(8, 8),
    up_channels=[[4, 8], [2, 4], [1, 2]],
    norm_layer=nn.InstanceNorm2d,
    use_dropout=False,
    loss_l1=dict(type='L1Loss'),
    loss_vgg=dict(type='VGGLoss'),
    loss_mask=dict(type='L1Loss'),
    pretrained=None
)

# dataset settings
dataset_type = 'CP_VTON'
data_root = 'data/VTON'
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    imgs_per_gpu=20,
    workers_per_gpu=4,
    drop_last=False,
    train=dict(
        GMM=dict(
            type=dataset_type,
            dataroot=os.path.join(data_root, 'vton_resize'),
            datamode='train',
            stage='GMM',
            data_list='train_pairs.txt',
            fine_height=256,
            fine_width=192,
            radius=5),
        TOM=dict(
            type=dataset_type,
            dataroot=os.path.join(data_root, 'vton_resize'),
            datamode='train',
            stage='TOM',
            data_list='train_pairs.txt',
            fine_height=256,
            fine_width=192,
            radius=5)),
    test=dict(
        GMM=dict(
            type=dataset_type,
            dataroot=os.path.join(data_root, 'vton_resize'),
            datamode='test',
            stage='GMM',
            data_list='test_pairs.txt',
            fine_height=256,
            fine_width=192,
            radius=5,
            save_dir=os.path.join(data_root, 'vton_resize', 'test')),
        TOM=dict(
            type=dataset_type,
            dataroot=os.path.join(data_root, 'vton_resize'),
            datamode='test',
            stage='TOM',
            data_list='test_pairs.txt',
            fine_height=256,
            fine_width=192,
            radius=5,
            save_dir=os.path.join(data_root, 'result'))))

# optimizer
optimizer = dict(type='Adam', lr=1e-4, betas=(0.5, 0.999))
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100000,
    warmup_ratio=0.1,
    step=[18, 20]
)

checkpoint_config = dict(interval=2)
log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

start_epoch = 0
total_epochs = 40
gpus = dict(train=[0], test=[0])
work_dir = 'checkpoint/CPVTON'
print_interval = 20  # interval to print information
save_interval = 2
init_weights_from = None
load_from = None
resume_from = None
checkpoint = None
workflow = [('train', total_epochs)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
