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

import pandas as pd
import numpy as np
from scipy.special import binom

import cv2




class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        plane = 512
        if model_name == 'resnet50':
            backbone = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
            plane = 2048 * 1 * 1
        elif model_name == 'resnet101':
            backbone = nn.Sequential(*list(models.resnet101(pretrained=pretrained).children())[:-2])
            plane = 2048 * 1 * 1
        else:
            backbone = None
        
        self.backbone = backbone
        self.mpool = nn.AdaptiveMaxPool2d((1, 1))
        self.apool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        
        feat = self.backbone(x)
        out = self.mpool(feat)
        out = out.view(out.size(0), -1)
        return out
    
class Dense(nn.Module):

    def __init__(self):
        super(Dense, self).__init__()

        self.bn = nn.BatchNorm1d(2048)
        self.fc = nn.Linear(2048, 600)

    def forward(self, inputs: torch.Tensor):

        out = self.fc(self.bn(inputs))
        out = nn.ELU(inplace=True)(out)
        return out