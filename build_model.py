import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

class GCN(nn.Module):
    def __init__(self,c,out_c,k=(7,7)): #out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0],1), padding =((k[0]-1)/2,0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1,k[0]), padding =(0,(k[0]-1)/2))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k[1]), padding =(0,(k[1]-1)/2))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1],1), padding =((k[1]-1)/2,0))
        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        
        x = x_l + x_r
        
        return x

class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
    
    def forward(self,x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        
        x = x + x_res
        
        return x

class FCN_GCN(nn.Module):
    def __init__(self, num_classes):     
        super(FCN_GCN, self).__init__()
        self.num_classes = num_classes #21 in paper

        resnet = models.resnet50(pretrained=True)
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        # self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1) #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048
        
        self.gcn1 = GCN(256,self.num_classes) #gcn_i after layer-1
        self.gcn2 = GCN(512,self.num_classes)
        self.gcn3 = GCN(1024,self.num_classes)
        self.gcn4 = GCN(2048,self.num_classes)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)
        self.br8 = BR(num_classes)
        self.br9 = BR(num_classes)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c,in_c,3,padding=1,bias=False),
            nn.BatchNorm2d(in_c/2),
            nn.ReLU(inplace=True),
            #nn.Dropout(.5),
            nn.Conv2d(in_c/2, self.num_classes, 1),

            )    

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.upsample(self.br5(gc_fm3 + gc_fm4), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.upsample(self.br6(gc_fm2 + gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm1 = F.upsample(self.br7(gc_fm1 + gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        out = F.upsample(self.br9(gc_fm1), input.size()[2:], mode='bilinear', align_corners=True)

        return out
