import os

# model settings
arch = 'vgg'
attribute_num = 1000
category_num = 48
img_size = (224, 224)
model = dict(
    type='GlobalPredictor',
    backbone=dict(type='Vgg', layer_setting='vgg16'),
    global_pool=dict(
        type='GlobalPooling',
        inplanes=(7, 7),
        pool_plane=(2, 2),
        inter_channels=[512, 4096],
        outchannels=4096),
    attr_predictor=dict(
        type='AttrPredictor', inchannels=4096, outchannels=attribute_num),
    loss_attr=dict(
        type='BCEWithLogitsLoss',
        ratio=1,
        weight=None,
        size_average=None,
        reduce=None,
        reduction='mean'),
    pretrained='checkpoint/vgg16.pth')

pooling = 'Global'

# dataset settings
dataset_type = 'Attr_Pred'
data_root = 'data/Attr_Predict'
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/train.txt'),
        label_file=os.path.join(data_root, 'Anno/train_attr.txt'),
        cate_file=os.path.join(data_root, 'Anno/train_cate.txt'),
        bbox_file=os.path.join(data_root, 'Anno/train_bbox.txt'),
        landmark_file=None,
        img_size=img_size),
    test=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/test.txt'),
        label_file=os.path.join(data_root, 'Anno/test_attr.txt'),
        cate_file=os.path.join(data_root, 'Anno/test_cate.txt'),
        bbox_file=os.path.join(data_root, 'Anno/test_bbox.txt'),
        landmark_file=None,
        attr_cloth_file=os.path.join(data_root, 'Anno/list_attr_cloth.txt'),
        img_size=img_size),
    val=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/val.txt'),
        label_file=os.path.join(data_root, 'Anno/val_attr.txt'),
        cate_file=os.path.join(data_root, 'Anno/val_cate.txt'),
        bbox_file=os.path.join(data_root, 'Anno/val_bbox.txt'),
        landmark_file=None,
        img_size=img_size))

# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9)
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

start_epoch = 0
total_epochs = 40
gpus = dict(train=[0], test=[0, 1, 2, 3])
work_dir = 'checkpoint/Predict/vgg/global'
print_interval = 20  # interval to print information
save_interval = 5
init_weights_from = 'checkpoint/vgg16.pth'
resume_from = None
load_from = None
workflow = [('train', total_epochs)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
