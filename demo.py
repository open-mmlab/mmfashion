import argparse
import io
import requests
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from data.datasets import data_loader
from data.data_processing import DataProcessing
from models.AttrNet import build_network
from utils.count_attr import count_attribute

from models.config import cfg


parser = argparse.ArgumentParser(description='Test an image')


parser.add_argument('--line_num','-l', type = int, help = 'read a line in test.txt')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch','-a', default = 'vgg16',
                    help = 'model architecture: ' +
                    '|'.join('vgg16') +
                    '(default:vgg16)')
parser.add_argument('--num_classes', default = 88, type = int,
                    help = 'number of classes in the dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--top_k', default=3, type=int, 
                    help='choose top k attributes')

args = parser.parse_args()
line_num = args.line_num

img_size = (224,224)
img_f = cfg.TEST_IMG_FILE
data = img_f.read()
lines = data.split('\n')
img_name = 'Img/'+lines[line_num]
print("read %s" %img_name)

bbox_f = cfg.TEST_BBOX_FILE
bboxes = np.loadtxt(bbox_f)
bbox = bboxes[line_num]
x1 = int(bbox[0]-10)
y1 = int(bbox[1]-10)
x2 = int(bbox[2]+10)
y2 = int(bbox[3]+10)
#print("bboxes are ")
#print(x1, y1, x2, y2)

label_f = cfg.TEST_LABEL_FILE 
labels = label_f.read().split('\n')
label = labels[line_num]
label = np.asarray(label.split(' '))
print("correct label :")
print(label)

landmarks_f = cfg.LANDMARKS_TEST_FILE
landmarks = landmarks_f.read().split('\n')
landmark = landmarks[line_num]
landmark = np.asarray(landmark.split(' '),dtype=np.float32).reshape(1,16)


# read the attribute name
attr = []
with open(cfg.ATTRIBUTES) as af:
    for line in af:
        parts = line.split()
        attr_name = parts[0]
        attr.append(attr_name)
af.close()

print("%d correct attributes are"% (len(label)-1))
for i,lab in enumerate(label):
    if lab == '1':
        print("%d th attribute %s"%(i,attr[i]))


# preprocess the image
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


def image_loader(img_name):
     img_pil = Image.open(img_name).crop(box=(x1,y1,x2,y2))
     # resize img
     img_pil.thumbnail(img_size, Image.ANTIALIAS)
     img_pil.save('demo_crop.jpg')
     
     img_tensor = transformations(img_pil)
     img_variable = Variable(img_tensor)
     image = img_variable.unsqueeze(0) # this is for VGG

     return image.cuda()


arch = args.arch

model = build_network()
model.cuda()

 
# load the model
print("=> Loading Network %s" % args.resume)
checkpoint = torch.load(args.resume)
cfg.start_epoch = checkpoint['epoch']

model.load_state_dict(checkpoint['state_dict'])

print("=> loaded checkpoint '{}' (epoch {})"
      .format(args.resume, checkpoint['epoch']))

cudnn.benchmark = True

# load image
image = image_loader(img_name)

model.eval()

img_landmarks = torch.autograd.Variable(torch.from_numpy(landmark))
output = model(x=image, landmarks=img_landmarks, single=True)

predicted = (F.sigmoid(output.data)).cpu().numpy()


print("predicted labels")
print(predicted)

        
index = np.argsort(predicted).reshape(1,88)
print(index.shape)

top1 = index[0][87]
top2 = index[0][86]
top3 = index[0][85]
top4 = index[0][84]
top5 = index[0][83]
print("there are %d predicted labels" % len(predicted))
print("top1 index : %d %s"% (top1,attr[top1]))
print("top2 index : %d %s" % (top2,attr[top2]))
print("top3 index : %d %s" % (top3,attr[top3]))
print("top4 index : %d %s" % (top3,attr[top4]))
print("top5 index : %d %s" % (top3,attr[top5]))
