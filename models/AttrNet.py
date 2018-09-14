import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from config import cfg
from layer_utils.ROIPool import ROIPooling
import numpy as np

class AttrNet(nn.Module):
    def __init__(self, num_classes=88, init_weights=True):
        super(AttrNet, self).__init__()
        
       # self.rois = rois
        # the first 4 shared conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # the first branch-- global image enter the 5th conv layer and fc1
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.second_branch_fc = nn.Sequential(
            nn.Linear(512*7*7, 512), #tbd
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

        self.fc2 = nn.Sequential(
        nn.Linear(8192, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        )
        self.fc3 = nn.Linear(4096, num_classes)

    # forward model, if single = True, forward a single image
    # if single = False, forward batch
    def forward(self, x, landmarks, single=False):
        # share first 4 conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # first branch-- continue to enter 5th conv layer

        first_branch_conv = self.conv5(x)

        first_branch_conv = first_branch_conv.view(first_branch_conv.size(0), -1)
        first_branch_out = self.fc1(first_branch_conv)
        
        # second branch -- roi pooling and fc layer

        if single is not False: # a single image(used for 'demo')
          #  print("landmark size", landmarks.size())
            landmarks = landmarks.view(-1) # change to 1D
         #   print("after view,", landmarks.size())
            landmark = landmarks.data.cpu().numpy().reshape(1,16)
            
            for k, cor in enumerate(landmark[0]):
                if k%2 ==0: #x
                    
                    x_cor = 14 * (cor/224.0)

                    y_cor = 14*(landmark[0][k+1]/224.0)
            
                else:
                    continue
                x1 = max(x_cor-3, 0)
                x2 = min(x_cor+4, 13)
                y1 = max(y_cor-3, 0)
                y2 = min(y_cor+4, 13)

                if x1==0:
                    x2=6
                if y1==0:
                    y2=6
                if x2 == 13:
                    x1=7
                if y2==13:
                    y1=7
                one_rois= np.array([x1,y1,x2,y2], dtype = np.int)
                if k ==0:
                    rois = one_rois
                    print(rois.shape)
                    print(rois.dtype)
                else:
                
                    rois = np.concatenate((rois, one_rois), axis = 0)
               
            rois = rois.reshape(1,32)
                
            for i,cor in enumerate(rois[0]):
                if i % 4 == 0: #x1
                    x1 = cor
                    y1 = rois[0][i+1]
                    x2 = rois[0][i+2]
                    y2 = rois[0][i+3]
                else:
                    continue
                
                if x2-x1 < 7:
                    x2 = x2+1
                if y2-y1 < 7:
                    y2= y2+1
        
                out_features = x[:,:, x1:x2, y1:y2].contiguous()
               
                out_feature_r = out_features.view(-1)
                second_branch_out = self.second_branch_fc(out_feature_r)
                if i ==0:
                    second_fc_output = second_branch_out
                else:
                    second_fc_output = torch.cat((second_fc_output, second_branch_out), 0)
                second_fc_output = second_fc_output.view(-1)
                roi_out =second_fc_output
                
            both_branch = torch.cat((first_branch_out, roi_out.view(1,4096)),0).view(-1)
       
        else: # batch

            roi_out = ROIPooling(x, landmarks, self.second_branch_fc) # n landmarks--n rois(512*6*6)
            
        # concat the output from the first and the second branch
            both_branch = torch.cat((first_branch_out, roi_out), 1)
               
        output = self.fc2(both_branch)
        output = self.fc3(output)
       
        return output

def initialize_weights(layers):
    if isinstance(layers, nn.Linear):
         nn.init.normal(layers.weight, 0, 0.01)
         nn.init.constant(layers.bias, 0)
    else:
        for m in layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.01)
                nn.init.constant(m.bias, 0)
                
def build_network():
    pretrained_vgg = torch.load(cfg.VGG16_BN_PATH)

    pretrained_list = list(pretrained_vgg.items())

    my_model = AttrNet( num_classes=cfg.num_classes)
    my_model_kvpair = my_model.state_dict()
    
    # load ImageNet-trained vgg16_bn weights
    count = 0
    
    # load all conv layers (conv1- conv5) and fc1 from pretrained ImageNet weights(79 parameters in total)
    for key, value in my_model_kvpair.items():
        if count <= 79:
            my_model_kvpair[key] = pretrained_list[count]
            
        count+=1

    # initialize fc2,fc3 and second_branch fc
    initialize_weights(my_model.second_branch_fc)
    initialize_weights(my_model.fc2)
    initialize_weights(my_model.fc3)

    return my_model
