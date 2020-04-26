import os

# model settings
arch = 'vgg'
retrieve = True
attribute_num = 303
id_num = 33881
img_size = (224, 224)
model = dict(
    type='RoIRetriever',
    backbone=dict(type='Vgg', layer_setting='vgg16'),
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
            reduction='mean'),
    ),
    pretrained='checkpoint/vgg16.pth')

pooling = 'RoI'

# extract feature or not
extract_feature = False

# dataset setting
dataset_type = 'ConsumerToShopDataset'
data_root = 'data/Consumer_to_shop'
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/train_consumer2shop.txt'),
        id_file=os.path.join(data_root, 'Anno/train_id.txt'),
        label_file=os.path.join(data_root, 'Anno/list_attr_items.txt'),
        bbox_file=os.path.join(data_root, 'Anno/list_bbox_train.txt'),
        landmark_file=os.path.join(data_root, 'Anno/list_landmarks_train.txt'),
        img_size=img_size,
        find_three=True),
    query=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/consumer.txt'),
        id_file=os.path.join(data_root, 'Anno/consumer_id.txt'),
        label_file=os.path.join(data_root, 'Anno/list_attr_items.txt'),
        bbox_file=os.path.join(data_root, 'Anno/list_bbox_consumer.txt'),
        landmark_file=os.path.join(data_root,
                                   'Anno/list_landmarks_consumer.txt'),
        img_size=img_size,
        find_three=True),
    gallery=dict(
        type=dataset_type,
        img_path=os.path.join(data_root, 'Img'),
        img_file=os.path.join(data_root, 'Anno/shop.txt'),
        id_file=os.path.join(data_root, 'Anno/shop_id.txt'),
        label_file=os.path.join(data_root, 'Anno/list_attr_items.txt'),
        bbox_file=os.path.join(data_root, 'Anno/list_bbox_shop.txt'),
        landmark_file=os.path.join(data_root, 'Anno/list_landmarks_shop.txt'),
        img_size=img_size,
        find_three=True))

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
total_epochs = 100
gpus = dict(train=[0, 1, 2], test=[0])
work_dir = 'checkpoint/Retrieve_Consumer_to_Shop/vgg/roi'
load_from = None
resume_from = None
init_weights_from = 'checkpoint/vgg16.pth'
workflow = [('train', 100)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
