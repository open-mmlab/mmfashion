import os


# model settings
arch = 'vgg'
class_num = 463
model = dict(
        type='RoIPredictor',
        backbone=dict(type='VggLayer'),
        global_pool=dict(type='GlobalPooling'),
        roi_pool=dict(type='RoIPooling'),
        concat=dict(type='Concat', num_classes=class_num),
        pretrained='checkpoint/vgg16.pth',
        )
pooling = 'RoI'

# dataset settings
dataset_type='In-shop'
data_root = 'datasets/In-shop'
img_norm = dict(
           mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data = dict(
           imgs_per_gpu=8, 
           workers_per_gpu=2,
           train = dict(
                   type='roi_dataset',
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/train_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/train_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/train_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/train_landmarks.txt'),
                   iuv_file=os.path.join(data_root, 'Anno/train_IUV.txt'),
                   img_scale=(224,224)
                   ),
           test = dict(
                   type='roi_dataset',
                   img_path = os.path.join(data_root, 'Img'),
                   img_file=os.path.join(data_root, 'Anno/test_img.txt'),
                   label_file=os.path.join(data_root, 'Anno/test_labels.txt'),
                   bbox_file=os.path.join(data_root, 'Anno/test_bbox.txt'),
                   landmark_file=os.path.join(data_root, 'Anno/test_landmarks.txt'),
                   iuv_file=os.path.join(data_root, 'Anno/test_IUV.txt'),
                   img_scale=(224,224)
                   )
           )

# optimizer
optimizer = dict(
             type='SGD', 
             lr=1e-3, 
             momentum=0.9)
# learning policy
lr_config = dict(
            policy='step',
            warmup_iters=10,
            warmup_ratio=0.1)

# loss 
loss_dict=dict(
           type='CrossEntropyLoss',
           weight=None,
           size_average=None,
           reduce=None,
           reduction='mean',
           use_sigmoid=True)
   
start_epoch=0
end_epoch=30
gpus=range(4)
work_dir = 'checkpoint/Predict'
print_interval=20 # interval to print information
save_interval=5
resume_from = None # 
load_from = 'checkpoint/Predict/vgg_RoI_epoch25.pth.tar'
workflow = [('train', 10)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
