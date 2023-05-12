import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ArcMargin(nn.Module):
    def __init__(self, in_feat, out_feat, s=30.0, m1=0.50):
        super(ArcMargin, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.s = s
        self.m1 = m1
        self.weight = Parameter(torch.FloatTensor(out_feat, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        #---------------------------- Margin Additional -----------------------------
        cos_m = math.cos(self.m1)
        sin_m = math.sin(self.m1)
        th = math.cos(math.pi - self.m1)
        mm = math.sin(math.pi - self.m1) * self.m1
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
        output *= self.s

        return self.s * cosine, output