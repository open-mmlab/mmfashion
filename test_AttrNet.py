import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

import pdb
import os
import argparse
import os
import shutil
import time
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from data.datasets import data_loader
from data.data_processing import DataProcessing
from models.AttrNet import build_network
from utils.count_attr import count_attribute

from models.config import cfg


attr = []
# read attribute index and attribute name
with open(cfg.ATTR_FILE) as f:
    for line in f:
        parts = line.split()
        attr_name = parts[0]
        attr.append(attr_name)
f.close()


# calculate top3 and top5 attribute accuracy for a batch
def cal_accuracy(outputs, targets):
    
    batch_size = cfg.batch_size
    # TP: true positive TN: true negative
    top3_TP = np.zeros(88) # 88 attributes
    top3_TN = np.zeros(88)
    top5_TP = np.zeros(88)
    top5_TN = np.zeros(88)
    
    for i in range(len(targets)): # the ith img
    
        target = targets[i].data
        output = outputs[i].data

        predicted = (F.sigmoid(output.cpu())).numpy()
        
        index = np.argsort(predicted)
       
        top3_index = []
        top5_index = []

        top1 = index[87]
        top3_index.append(top1)
        top5_index.append(top1)

        top2 = index[86]
        top3_index.append(top2)
        top5_index.append(top2)
        
        top3 = index[85]
        top3_index.append(top3)
        top5_index.append(top3)
        
        top4 = index[84]
        top5_index.append(top4)

        top5 = index[83]
        top5_index.append(top5)
        

        # rule out the images that don't have any attribute
        if sum(target) == 0:
            continue # move to the next image
        
        for attr_idx,a in enumerate(target):
            if a == 1: # positive
                if attr_idx in top3_index:
                    top3_TP[attr_idx]+=1

                if attr_idx in top5_index:
                    top5_TP[attr_idx] += 1
                    
            if a == 0: # negative
                if attr_idx not in top3_index:
                    top3_TN[attr_idx] +=1
                if attr_idx not in top5_index:
                    top5_TN[attr_idx] +=1

        
    return top3_TP, top3_TN, top5_TP, top5_TN


def main():
        
    arch = cfg.arch

    model = build_network()
    model.cuda()
    print model

    # load the model
    print("=> Loading Network %s" % cfg.resume)
    checkpoint = torch.load(cfg.resume)
     
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(cfg.resume, checkpoint['epoch']))

    print model

    cudnn.benchmark = False

    test_loader = data_loader( BatchSize=cfg.batch_size,
                               NumWorkers = cfg.num_workers).test_loader
    if cfg.loss == 'MSE':
        criterion = nn.MSELoss()
    elif cfg.loss == 'MLS':
        criterion = nn.MultiLabelMarginLoss()
    else:
        criterion =  nn.BCELoss()

    optimizer =  torch.optim.SGD(model.parameters(), cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    print("test data_loader are ready!")
    
    # test mode
    model.eval()

    # test an image
    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    sum_batch = 0
    # the accuracy for each attribute 
    top3_accuracy = np.zeros(88)
    top5_accuracy = np.zeros(88)

    top3_TP = np.zeros(88)
    top5_TP = np.zeros(88)

    top3_TN = np.zeros(88)
    top5_TN = np.zeros(88)
    
    for iter, test_data in enumerate(test_loader):
        sum_batch += 1
        test_inputs, test_labels, landmarks= test_data # not use landmarks while testing
        test_inputs, test_labels, landmarks =  torch.autograd.Variable(test_inputs.cuda(), volatile=True).float(), torch.autograd.Variable(test_labels.cuda(),volatile=True).float(), torch.autograd.Variable(landmarks.cuda(),volatile=True)

        test_outputs = model(test_inputs, landmarks, single=False)
        top3_TP_batch = np.zeros(88)
        top5_TP_batch = np.zeros(88)

        top3_TN_batch = np.zeros(88)
        top5_TN_batch = np.zeros(88)
        # calculate accuracy for a batch
        top3_TP_batch, top5_TN_batch, top5_TP_batch, top5_TN_batch = cal_accuracy( test_outputs, test_labels)

        for tp_i, tp in enumerate(top3_TP_batch):
            top3_TP[tp_i] += top3_TP_batch[tp_i]
            top3_TN[tp_i] += top3_TN_batch[tp_i]
            top5_TP[tp_i] += top5_TP_batch[tp_i]
            top5_TN[tp_i] += top5_TN_batch[tp_i]
                    
    attr_count, image_count = count_attribute(cfg.TEST_LABEL_FILE) # the ground truth
    
    for acc_i, acc in enumerate(top3_TP):
        top3_accuracy[acc_i] = 100* (float(top3_TP[acc_i]+top3_TN[acc_i])/float(image_count))
    for acc_i,acc in enumerate(top5_TP):
        top5_accuracy[acc_i] = 100*(float(top5_TP[acc_i]+top5_TN[acc_i])/float(image_count))
        
    print("attribute count")
    print(attr_count)
    
    print("================t o p 3======================")
    print("top3 accuracy for each attribute is :")
    for i,acc in enumerate(top3_accuracy):
        print("%d th attribute %s : %.4f" %(i, attr[i], acc))
    print('\n')
    print('\n')
    for i,count in enumerate(top3_TP):
        print("%d th attribute %s : %d" %(i, attr[i], count))
    print("top3 average accuracy : %.4f"%(float(sum(top3_accuracy))/len(top3_accuracy)))
    
    print('\n')
    print("================t o p 5=======================")
    print("top5 accuracy for each attribute is :")
    for i,acc in enumerate(top5_accuracy):
        #print("%d th attribute %s : %.4f" %(i, attr[i], acc))
        print("%.4f" %acc)
    print('\n')
    print('\n')
    for i,count in enumerate(top5_TP):
        print("%d th attribute %s : %d" %(i, attr[i], count))
    print("top5 average accuracy : %.4f"%(float(sum(top5_accuracy))/len(top5_accuracy)))

    

if __name__ == '__main__':
    main()
    
        
        

        


