import os

# model settings
arch = 'vgg'
attribute_num = 26  # num of attributes
category_num = 50  # num of categories
img_size = (224, 224)

model = dict(
    type='RoIAttrCatePredictor',
    backbone=dict(type='Vgg', layer_setting='vgg16'),
    global_pool=dict(
        type='GlobalPooling',
        inplanes=(7, 7),
        pool_plane=(2, 2),
        inter_channels=[512, 1024],
        outchannels=1024),
    roi_pool=dict(
        type='RoIPooling',
        pool_plane=(2, 2),
        inter_channels=512,
        outchannels=1024,
        crop_size=7,
        img_size=img_size,
        num_lms=8),
    concat=dict(type='Concat', inchannels=2 * 1024, outchannels=1024),
    attr_predictor=dict(
        type='AttrPredictor',
        inchannels=1024,
        outchannels=attribute_num,
        loss_attr=dict(
            type='BCEWithLogitsLoss',
            ratio=1,
            weight=None,
            size_average=None,
            reduce=None,
            reduction='mean')),
    cate_predictor=dict(
        type='CatePredictor',
        inchannels=1024,
        outchannels=category_num,
        loss_cate=dict(
            type='CELoss',
            ratio=1,
            weight=None,
            reduction='mean')),
    pretrained='checkpoint/vgg16.pth')

pooling = 'RoI'

# dataset settings
dataset_type = 'Attr_Pred'
data_root = 'data/Attr_Predict'
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno_fine/train.txt'),
        label_file=os.path.join(data_root, 'Anno_fine/train_attr.txt'),
        cate_file=os.path.join(data_root, 'Anno_fine/train_cate.txt'),
        bbox_file=os.path.join(data_root, 'Anno_fine/train_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno_fine/train_landmarks.txt'),
        img_size=img_size),
    test=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno_fine/test.txt'),
        label_file=os.path.join(data_root, 'Anno_fine/test_attr.txt'),
        cate_file=os.path.join(data_root, 'Anno_fine/test_cate.txt'),
        bbox_file=os.path.join(data_root, 'Anno_fine/test_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno_fine/test_landmarks.txt'),
        attr_cloth_file=os.path.join(data_root, 'Anno_fine/list_attr_cloth.txt'),
        cate_cloth_file=os.path.join(data_root, 'Anno_fine/list_category_cloth.txt'),
        img_size=img_size),
    val=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno_fine/val.txt'),
        label_file=os.path.join(data_root, 'Anno_fine/val_attr.txt'),
        cate_file=os.path.join(data_root, 'Anno_fine/val_cate.txt'),
        bbox_file=os.path.join(data_root, 'Anno_fine/val_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno_fine/val_landmarks.txt'),
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
total_epochs = 50
gpus = dict(train=[0, 1], test=[0])
work_dir = 'checkpoint/CateAttrPredict/vgg/roi'
print_interval = 20  # interval to print information
save_interval = 5
init_weights_from = None
load_from = None
resume_from = None
workflow = [('train', total_epochs)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
