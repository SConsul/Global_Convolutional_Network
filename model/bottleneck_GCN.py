import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

class bottleneck_GCN(nn.Module):
    def __init__(self, m, c, k):
        super(bottleneck_GCN, self).__init__()
        self.bn_m = nn.BatchNorm2d(m)
        self.bn_c = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_l1 = nn.Conv2d(c, m, kernel_size=(k,1), padding =((k-1)/2,0))
        self.conv_l2 = nn.Conv2d(m, m, kernel_size=(1,k), padding =(0,(k-1)/2))
        self.conv_r1 = nn.Conv2d(c, m, kernel_size=(1,k), padding =((k-1)/2,0))
        self.conv_r2 = nn.Conv2d(m, m, kernel_size=(k,1), padding =(0,(k-1)/2))
        self.conv_f = n.Con2vd(m, c, kernel_size=1, padding=0)
        
    def forward(self,x):
        x_res_l = self.conv_l1(x)
        x_res_l = self.bn_m(x_res_l)
        x_res_l = self.relu(x_res_l)
        x_res_l = self.conv_l2(x_res_l)
        x_res_l = self.bn_m(x_res_l)
        x_res_l = self.relu(x_res_l)
        
        x_res_r = self.conv_r1(x)
        x_res_r = self.bn_m(x_res_r)
        x_res_r = self.relu(x_res_r)
        x_res_r = self.conv_r2(x_res_r)
        x_res_r = self.bn_m(x_res_r)
        x_res_r = self.relu(x_res_r)
        
        x_res = x_res_l + x_res_r
        x_res = self.conv_f(x_res)
        x_res = self.bn_c(x_res)
        
        x = x + x_res
        
        return x