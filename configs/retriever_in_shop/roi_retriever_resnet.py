import os

# model settings
arch = 'resnet'
retrieve = True
attribute_num = 463
id_num = 7982
img_size = (224, 224)
model = dict(
    type='RoIRetriever',
    backbone=dict(type='ResNet', setting='resnet50'),
    global_pool=dict(
        type='GlobalPooling',
        inplanes=(7, 7),
        pool_plane=(2, 2),
        inter_channels=[2048, 4096],
        outchannels=4096),
    roi_pool=dict(
        type='RoIPooling',
        pool_plane=(2, 2),
        inter_channels=2048,
        outchannels=4096,
        crop_size=7,
        img_size=img_size,
        num_lms=8),
    concat=dict(type='Concat', inchannels=2 * 4096, outchannels=4096),
    embed_extractor=dict(
        type='EmbedExtractor',
        inchannels=4096,
        inter_channels=[256, id_num],
        loss_id=dict(type='CELoss', ratio=1),
        loss_triplet=dict(type='TripletLoss', method='cosine', margin=0.)),
    attr_predictor=dict(
        type='AttrPredictor',
        inchannels=4096,
        outchannels=attribute_num,
        loss_attr=dict(
            type='BCEWithLogitsLoss',
            ratio=1,
            weight=None,
            size_average=None,
            reduce=None,
            reduction='mean')),
    pretrained='checkpoint/resnet50.pth')

pooling = 'RoI'

# extract feature or nor
extract_feature = True

# dataset settings
dataset_type = 'InShopDataset'
data_root = 'data/In-shop'
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/train_img.txt'),
        label_file=os.path.join(data_root, 'Anno/train_labels.txt'),
        id_file=os.path.join(data_root, 'Anno/train_id.txt'),
        bbox_file=os.path.join(data_root, 'Anno/train_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/train_landmarks.txt'),
        img_size=img_size,
        roi_plane_size=7,
        retrieve=retrieve,
        find_three=True),
    query=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/query_img.txt'),
        label_file=os.path.join(data_root, 'Anno/query_labels.txt'),
        id_file=os.path.join(data_root, 'Anno/query_id.txt'),
        bbox_file=os.path.join(data_root, 'Anno/query_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/query_landmarks.txt'),
        img_size=img_size,
        roi_plane_size=7,
        retrieve=retrieve,
        find_three=retrieve),
    gallery=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/gallery_img.txt'),
        label_file=os.path.join(data_root, 'Anno/gallery_labels.txt'),
        id_file=os.path.join(data_root, 'Anno/gallery_id.txt'),
        bbox_file=os.path.join(data_root, 'Anno/gallery_bbox.txt'),
        landmark_file=os.path.join(data_root, 'Anno/gallery_landmarks.txt'),
        img_size=img_size,
        roi_plane_size=7,
        retrieve=retrieve,
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
    step=[20, 40])

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

start_epoch = 0
total_epochs = 150
gpus = dict(train=[0, 1, 2, 3], test=[0])
work_dir = 'checkpoint/Retrieve/resnet/roi/with_attr'
print_interval = 20
resume_from = None
load_from = 'checkpoint/Retrieve/resnet/roi/with_attr/latest.pth'
init_weights_from = 'checkpoint/resnet50.pth'
workflow = [('train', 100)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
