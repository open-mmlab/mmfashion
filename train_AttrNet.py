import argparse
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from collections import OrderedDict
import pdb

from data.data_loader import data_loader
from data.data_processing import DataProcessing 

from models.config import cfg
from models.AttrNet import build_network
from models.loss.WeightedBCELoss import *

def main():

    # create model
    print("=> creating model based on '{}'".format('fashion detection network'))

    # 88 attributes, 88 classes

    # build the network
    model = build_network()
    
    print model

    # use multiple gpus
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("now we are using %d gpus" %torch.cuda.device_count())
    model.cuda()
    
    optimizer =  torch.optim.SGD(model.parameters(), cfg.lr,
                                 momentum = cfg.momentum,
                                 weight_decay=cfg.weight_decay)

    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)            
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))
    
    train_loader = data_loader( BatchSize=cfg.batch_size,
                               NumWorkers = cfg.num_workers).train_loader
    test_loader = data_loader( BatchSize=cfg.batch_size,
                               NumWorkers = cfg.num_workers).test_loader

    
    # define loss function (criterion) and optimizer
    # this loss combines a Sigmoid layer and the BCELoss in one single class
    if cfg.loss == 'MSE':
        criterion =  nn.MSELoss()
    elif cfg.loss == 'MLS':
        criterion = nn.MultiLabelMarginLoss()
    else:
        criterion = WeightedBCELoss( reduce=False, size_average=False)

        
    if cfg.opt=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.lr,
                                     weight_decay = cfg.weight_decay)
    else:
        optimizer =  torch.optim.SGD(model.parameters(), cfg.lr,
                                     momentum = cfg.momentum,
                                     weight_decay=cfg.weight_decay)
            
    print("train data_loader are ready!")

    for epoch in range(cfg.start_epoch, cfg.epochs):
        adjust_learning_rate(optimizer,epoch)

        # train for one epoch
        train_model(train_loader, model, criterion, optimizer, epoch, cfg.epochs)

        # save checkpoint for every epoch
        logging.info("saving the latest model(epoch %d, cfg.epochs %d)"
                     %(epoch, cfg.epochs))
        checkpoint_name = cfg.save_model_dir + cfg.arch + '_AttrNet_epoch%d'%epoch+'.tar'
        torch.save({'epoch': epoch+1,
                    'arch': cfg.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
        }, checkpoint_name)
                    
        
def train_model(train_loader, model, criterion, optimizer, epoch, epochs):

    model.train()

    for iter, traindata in enumerate(train_loader):
        train_inputs, train_labels, train_landmarks = traindata        
        train_inputs, train_labels, train_landmarks = torch.autograd.Variable(train_inputs.cuda(async=True),).float(), torch.autograd.Variable(train_labels.cuda(async=True)).float(), torch.autograd.Variable(train_landmarks.cuda(async=True)).float()

        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        train_outputs = model(train_inputs, train_landmarks)
        if cfg.loss == 'BCE':
            train_outputs = F.sigmoid(train_outputs)
        
        loss = criterion(train_outputs, train_labels)
        loss.backward()        
        optimizer.step()

        print("loss.data[0]", loss.data[0])
        print('Training Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.4f' %
              (iter, epoch, epochs, loss.data[0]))

        # free some memory here
#        del loss
#        if iter % 2000 == 0:
#            checkpoint_name = cfg.save_model_dir + cfg.arch + '_AttrNet_epoch%d'%epoch+ 'iter_%d'%iter+'.tar#'
#            torch.save({'epoch': epoch+1,
#                        'arch': cfg.arch,
#                        'state_dict': model.state_dict(),
#                        'optimizer' : optimizer.state_dict(),
#            }, checkpoint_name)

        

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cfg.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
                        
