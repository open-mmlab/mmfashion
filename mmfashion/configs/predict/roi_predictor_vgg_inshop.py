import os

# model settings
arch = 'vgg'
retrieve = False
attribute_num = 463
id_num = 7982
img_size = (224, 224)
retrieve=False
model = dict(
    type='RoIPredictor',
    backbone=dict(type='Vgg'),
    global_pool=dict(
        type='GlobalPooling',
        inplanes=(7, 7),
        pool_plane=(2, 2),
        inter_channels=[512, 4096],
        outchannels=4096),
    roi_pool=dict(
        type='RoIPooling',
        pool_plane=(2, 2),
        inter_channels=512,
        outchannels=4096,
        crop_size=7,
        img_size=img_size,
        num_lms=8),
    concat=dict(
        type='Concat',
        inchannels=2 * 4096,
        inter_channels=[4096,1024],
        num_attr=attribute_num,
        num_cate=id_num,
        retrieve=retrieve),
    loss_attr=dict(
        type='BCEWithLogitsLoss',
        weight=None,
        size_average=None,
        reduce=None,
        reduction='mean'),
    loss_cate=dict(
        type='CELoss'),
    pretrained='checkpoint/vgg16.pth')

pooling = 'RoI'

# dataset settings
dataset_type = 'In-shop'
data_root = '../data/In-shop'
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/train_img.txt'),
        id_file=os.path.join(data_root, 'Anno/train_id.txt'),
        label_file=os.path.join(data_root, 'Anno/train_labels.txt'),
        bbox_file=os.path.join(data_root, 'Anno/train_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/train_landmarks.txt'),
        img_size=img_size,
        roi_plane_size=7,
        find_three=retrieve
    ),
    test=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/test_img.txt'),
        id_file=os.path.join(data_root, 'Anno/test_id.txt'),
        label_file=os.path.join(data_root, 'Anno/test_labels.txt'),
        bbox_file=os.path.join(data_root, 'Anno/test_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/test_landmarks.txt'),
        img_size=img_size,
        roi_plane_size=7,
        find_three=retrieve),
    val=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/val_img.txt'),
        id_file=os.path.join(data_root, 'Anno/val_id.txt'),
        label_file=os.path.join(data_root, 'Anno/val_labels.txt'),
        bbox_file=os.path.join(data_root, 'Anno/val_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/val_landmarks.txt'),
        img_size=img_size,
        roi_plane_size=7,
        find_three=retrieve))

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
gpus = dict(train=[0, 1, 2, 7], test=[0])
work_dir = 'checkpoint/Predict/vgg'
print_interval = 20  # interval to print information
save_interval = 5
init_weights_from = 'checkpoint/vgg16.pth'
resume_from = None
checkpoint = 'checkpoint/Retrieve/vgg/latest.pth'
workflow = [('train', 40)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
