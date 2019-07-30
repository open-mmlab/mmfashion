import os


# model settings
arch = 'vgg'
retrieve=True
class_num = 463
img_size=(224,224)
model = dict(
        type='RoIRetriever',
        backbone=dict(type='Vgg'),
        global_pool=dict(type='GlobalPooling',
                         inplanes=(7,7),
                         pool_plane=(2,2),
                         inter_plane=512*7*7,
                         outplanes=4096),
        roi_pool=dict(type='RoIPooling',
                      pool_plane=(2,2),
                      inter_plane=512,
                      outplanes=4096,
                      crop_size=7,
                      img_size=img_size,
                      num_lms=8),
        concat=dict(type='Concat', 
                    inplanes=2*4096,
                    inter_plane=4096,
                    num_classes=class_num,
                    retrieve=retrieve),
        loss_cls = dict(
                   type='BCEWithLogitsLoss',
                   weight=None,
                   size_average=None,
                   reduce=None,
                   reduction='mean'),
        loss_retrieve=dict(
             type='TripletLoss',
             margin=1.0,
             use_sigmoid=True,
             size_average=True
             ),
        pretrained='checkpoint/vgg16.pth'
        )

pooling = 'RoI'

# dataset settings
dataset_type='InShopDataset'
data_root = 'datasets/In-shop'
img_norm = dict(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225])

data = dict(
           imgs_per_gpu=8,
           workers_per_gpu=4,
           train = dict(
                   type=dataset_type,
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/train_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/train_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/train_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/train_landmarks.txt'),
                   img_size=(224,224),
                   retrieve=retrieve,
                   find_three=True
                   ),
           query = dict(
                   type=dataset_type,
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/query_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/query_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/query_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/query_landmarks.txt'),
                   img_size=(224,224),
                   retrieve=retrieve,
                   find_three=True,
                   idx2id = os.path.join(data_root, 'Anno/query_idx2id.txt')
                   ),
           gallery = dict(
                   type=dataset_type,
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/gallery_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/gallery_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/gallery_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/gallery_landmarks.txt'),
                   img_size=(224,224),
                   retrieve=retrieve,
                   find_three=True,
                   idx2id = os.path.join(data_root, 'Anno/gallery_idx2id.txt')
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
            step=[20,40])

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])


start_epoch=0
total_epochs=100
gpus=dict(train=[0,1,2,3],
          test=[0])
work_dir = 'checkpoint/Retrieve/vgg'
print_interval=20 # interval to print information
resume_from = None
load_from = None
init_weights_from = 'checkpoint/Predict/vgg/attr_pred/latest.pth'
workflow = [('train', 100)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
