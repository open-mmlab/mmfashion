import os

# model setting
arch = 'vgg'
landmark_num = 8
img_size = (224, 224)

model = dict(
    type='LandmarkDetector',
    backbone=dict(type='Vgg', layer_setting='vgg16'),
    global_pool=dict(
        type='GlobalPooling',
        inplanes=(7, 7),
        pool_plane=(2, 2),
        inter_channels=[512, 4096],
        outchannels=4096),
    landmark_feature_extractor=dict(
        type='LandmarkFeatureExtractor',
        inchannels=4096,
        feature_dim=256,
        landmarks=landmark_num),
    visibility_classifier=dict(
        type='VisibilityClassifier',
        inchannels=256,
        outchannels=2,
        landmark_num=landmark_num,
        loss_vis=dict(
            type='BCEWithLogitsLoss',
            ratio=1,
            weight=None,
            size_average=None,
            reduce=None,
            reduction='none')),
    landmark_regression=dict(
        type='LandmarkRegression',
        inchannels=256,
        outchannels=2,
        landmark_num=landmark_num,
        loss_regress=dict(type='MSELoss', ratio=1, reduction='none')),
    pretrained='checkpoint/vgg16.pth')

# dataset settings
dataset_type = 'Landmark_Detect'
data_root = 'data/Landmark_Detect'
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/train.txt'),
        bbox_file=os.path.join(data_root, 'Anno/train_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/train_landmarks.txt'),
        img_size=img_size),
    test=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/test.txt'),
        bbox_file=os.path.join(data_root, 'Anno/test_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/test_landmarks.txt'),
        img_size=img_size),
    val=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/val.txt'),
        bbox_file=os.path.join(data_root, 'Anno/val_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/val_landmarks.txt'),
        img_size=img_size))

# optimizer
optimizer = dict(type='SGD', lr=1e-6, momentum=0.9)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[10, 20])

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

start_epoch = 101
total_epochs = 150
gpus = dict(train=[0, 1, 2, 3], test=[0, 1, 2, 3])
work_dir = 'checkpoint/LandmarkDetect/vgg/global'
print_interval = 20  # interval to print information
save_interval = 10
init_weights_from = 'checkpoint/vgg16.pth'
load_from = None
resume_from = None
checkpoint = None
workflow = [('train', total_epochs)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
