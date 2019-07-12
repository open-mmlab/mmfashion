import os


# model settings
arch = 'resnet'
retrieve=False
class_num = 463
img_size=(224,224)
model = dict(
        type='RoIPredictor',
        backbone=dict(type='ResNet'),
        global_pool=dict(type='GlobalPooling',
                         inplanes=(7,7),
                         pool_plane=(2,2),
                         inter_plane=2048*7*7,
                         outplanes=2048),
        roi_pool=dict(type='RoIPooling',
                      pool_plane=(2,2),
                      inter_plane=2048,
                      outplanes=2048,
                      crop_size=7,
                      img_size=img_size,
                      num_lms=8),
        concat=dict(type='Concat',
                    inplanes=2*2048,
                    inter_plane=2048,
                    num_classes=class_num,
                    retrieve=retrieve),
        loss=dict(
           type='BCEWithLogitsLoss',
           weight=None,
           size_average=None,
           reduce=None,
           reduction='mean'),
        pretrained='checkpoint/resnet50.pth',
        )

pooling = 'RoI'

# dataset settings
dataset_type='In-shop'
data_root = 'datasets/In-shop'
img_norm = dict(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225])

data = dict(
           imgs_per_gpu=1,
           workers_per_gpu=2,
           train = dict(
                   type=dataset_type,
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/train_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/train_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/train_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/train_landmarks.txt'),
                   img_scale=img_size,
                   find_three=retrieve # if retrieve, then find three items: anchor, pos, neg
                   ),
           test = dict(
                   type=dataset_type,
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/test_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/test_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/test_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/test_landmarks.txt'),
                   img_scale=img_size,
                   find_three=retrieve
                   ),
           val = dict(
                   type=dataset_type,
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/val_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/val_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/val_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/val_landmarks.txt'),
                   img_scale=img_size,
                   find_three=retrieve
                   )
           )

# optimizer
optimizer = dict(
             type='SGD',
             lr=1e-3,
             momentum=0.9)
optimizer_config = dict()

# learning policy
lr_config = dict(
            policy='step',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=0.1,
            step=[10,20])

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

start_epoch=0
total_epochs=40
gpus=dict(train=4,
          test=4)
work_dir = 'checkpoint/Predict'
print_interval=20 # interval to print information
save_interval=5
resume_from = 'checkpoint/resnet50.pth'  
load_from = None#'checkpoint/Predict/vgg_RoI_epoch25.pth.tar'
workflow = [('train', 40)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
                  
