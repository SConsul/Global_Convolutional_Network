import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

from GCN import GCN
from BR import BR

class FCN_GCN(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(FCN_GCN, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048
        
        self.gcn1 = GCN(256,self.num_classes,55) #gcn_i after layer-1
        self.gnc2 = GCN(512,self.num_classes,27)
        self.gcn3 = GCN(1024,self.num_classes,13)
        self.gcn4 = GCN(2048,self.num_classes,7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)
        self.br8 = BR(num_classes)
        self.br9 = BR(num_classes)

    def _classifier(self, c):
        return nn.Sequential(
            nn.Conv2d(c,c,3,padding=1,bias=False),
            

            )    
        
    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        