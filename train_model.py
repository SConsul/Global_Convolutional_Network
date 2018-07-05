import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from build_model import FCN_GCN
import os
import csv

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class SoftInvDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftInvDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = 1 - logits.view(-1)
        tflat = 1 - targets.view(-1)
        intersection = (iflat * tflat).sum()
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))
    
img_size = (1024,1024)    
transformations_train = transforms.Compose([transforms.Resize(img_size),
                                      transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
    
transformations_test = transforms.Compose([transforms.Resize(img_size),
                                      transforms.ToTensor()])     
                                      
    
  
from data_loader import LungSeg
from data_loader import LungSegTest
train_set = LungSeg(transforms = transformations_train)  
test_set = LungSegTest(transforms = transformations_test)  
batch_size = 1 
num_epochs = 30
    
class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

def train():
    cuda = torch.cuda.is_available()
    net = FCN_GCN(1)
    net.load_state_dict(torch.load('cp.pth'))

    if cuda:
        net = net.cuda()
    criterion1 = nn.BCEWithLogitsLoss().cuda()
    criterion2 = SoftDiceLoss().cuda()
    criterion3 = SoftInvDiceLoss().cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=4e-5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.5)

    print("preparing training data ...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("done ...")

    test_set = LungSegTest(transforms = transformations_test) 
    test_loader = DataLoader(test_set, batch_size=batch_size)
    for epoch in range(num_epochs):
        train_loss = Average()
        net.train()
        
        #scheduler.step()

        for i, (images, masks) in enumerate(train_loader):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = 0.4*criterion1(outputs, masks) + 0.4*criterion2(outputs, masks)  + 0.2*criterion3(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), images.size(0))

        val_loss = Average()
        val_loss_dice = Average()
        net.eval()
        for images, masks in test_loader:
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            outputs = net(images)
            vloss = 0.4*criterion1(outputs, masks) + 0.4*criterion2(outputs, masks)  + 0.2*criterion3(outputs, masks)
            vloss_dice = criterion2(outputs, masks)
            val_loss.update(vloss.item(), images.size(0))
            val_loss_dice.update(vloss_dice.item(), images.size(0))

        print("Epoch {}/{}, Loss: {}, Validation Loss: {}, Validation Dice Loss: {}".format(epoch+1,num_epochs, train_loss.avg, val_loss.avg, val_loss_dice.avg))
       
        torch.save(net.state_dict(), 'Weights_221/cp_{}_{}.pth'.format(epoch+1, val_loss_dice.avg))    
            
    return net

def test(model):
    model.eval()



if __name__ == "__main__":
    train()
