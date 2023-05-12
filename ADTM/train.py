import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, ExponentialLR

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
from scipy.special import binom

import cv2
import os
import math
import time
import datetime
import warnings

from loss import *
from model import *
from datasets import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.0001
base_lr = 0.01
weight_decay = 1e-4
num_classes = 200

Model = BaseModel('resnet50', pretrained=True).to(device)
Dense = Dense().to(device)
Arcface = ArcMargin(600, num_classes, s=64, m1=0.5).to(device) # 78.62
criterion = nn.CrossEntropyLoss()

# optimizer
base_opt = torch.optim.SGD(Model.parameters(), 
                            lr=lr, 
                            momentum=0.9,
                            weight_decay=weight_decay, 
                            nesterov=True)

optimizer = torch.optim.SGD([{'params': Dense.parameters()}, {'params': Arcface.parameters()}], 
                            lr=base_lr, 
                            momentum=0.9,
                            weight_decay=weight_decay, 
                            nesterov=True)

# optimization scheduler
basesc = torch.optim.lr_scheduler.MultiStepLR(base_opt, milestones=[60], gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 75], gamma=0.1)




## ---- Model Training ---

epochs = 100
save_model_path = 'Checkpoint'
steps = 0
running_loss = 0

print('Start fine-tuning...')
best_acc = 0.
best_epoch = None

for epoch in range(1, epochs+1):
    
    start_time = time.time()
    for idx, data in enumerate(train_loader):
        steps += 1
        
        # Move input and label tensors to the default device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        base_opt.zero_grad(), optimizer.zero_grad()
        
        output = Model(inputs)
        output = Dense(output)
        _, output = Arcface(output, labels)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        base_opt.step()

        running_loss += loss.item()
    stop_time = time.time()
    print('Epoch {}/{} and used time: {:.2f} sec.'.format(epoch, epochs, stop_time - start_time))
    
    Model.eval(), Arcface.eval(), Dense.eval()
    for name, loader in [("train", train_loader), ("test", test_loader)]:
        _loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in loader:
                
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)
                
                output = Model(imgs)
                output = Dense(output)
                cosine, output = Arcface(output, labels)
                loss = criterion(output, labels)
                _loss += loss.item()
                
                result = F.softmax(cosine, dim=1)
                _, predicted = torch.max(result, dim=1)
                
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
            _acc = 100 * correct  / total
            _loss = _loss / len(loader)
            
        print('{} loss: {:.4f}    {} accuracy: {:.4f}'.format(name, _loss, name, _acc))
    print()
    
    running_loss = 0
    Model.train(), Arcface.train(), Dense.train()
    scheduler.step()
    basesc.step()
    
    if _acc > best_acc:
        best_acc = _acc
        best_epoch = epoch

print('After the training, the end of the epoch {}, the highest accuracy is {:.2f}'.format(best_epoch, best_acc))