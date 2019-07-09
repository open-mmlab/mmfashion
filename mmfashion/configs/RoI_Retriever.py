import os


# model settings
arch = 'vgg'
retrieve=True
class_num = 463
model = dict(
        type='RoIRetriever',
        backbone=dict(type='VggLayer'),
        global_pool=dict(type='GlobalPooling'),
        roi_pool=dict(type='RoIPooling'),
        concat=dict(type='Concat', 
                    num_classes=class_num,
                    retrieve=retrieve),
        loss=dict(
             type='TripletLoss',
             margin=1.0,
             use_sigmoid=True,
             size_average=True
             ),
        pretrained='checkpoint/vgg16.pth',
        )
pooling = 'RoI'

# dataset settings
dataset_type='In-shop'
data_root = 'datasets/In-shop'
img_norm = dict(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225])

data = dict(
           imgs_per_gpu=8, 
           workers_per_gpu=2,
           train = dict(
                   type=dataset_type,
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/train_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/train_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/train_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/train_landmarks.txt'),
                   img_scale=(224,224),
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
                   img_scale=(224,224),
                   retrieve=retrieve,
                   find_three=False
                   ),
           gallery = dict(
                   type=dataset_type,
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/gallery_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/gallery_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/gallery_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/gallery_landmarks.txt'),
                   img_scale=(224,224),
                   retrieve=retrieve,
                   find_three=False
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
gpus=dict(train=4,
          test=1)
work_dir = 'checkpoint/Retriever'
print_interval=20 # interval to print information
save_interval=5
resume_from = 'checkpoint/Predict/vgg_RoI_epoch25.pth.tar'  
load_from = 'checkpoint/Predict/vgg_RoI_epoch25.pth.tar'
workflow = [('train', 100)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
