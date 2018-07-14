import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
    
    def forward(self,x):
        x_res = x
#         x_res = self.bn(x)
#         x_res = self.relu(x_res)
        x_res = self.conv1(x_res)
#         x_res = self.bn(x_res)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        
        x = x + x_res
        
        return x
