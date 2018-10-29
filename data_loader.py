
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image
import os
import numpy as np


class LungSeg(Dataset):
    def __init__(self, path='Image/Train/Images', transforms=None):
        self.path = path
        self.list = os.listdir(self.path)
        
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        image_path = 'Image/Train/Images/'
        mask_path = 'Mask/Train/Images/'
        image = Image.open(image_path+self.list[index])
        image = image.convert('RGB')
        mask = Image.open(mask_path+self.list[index])
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (image, mask)

    def __len__(self):
        return len(self.list) # of how many data(images?) you have

    
class LungSegTest(Dataset):
    def __init__(self, path='Image/Test/Images', transforms=None):
        self.path = path
        self.list = os.listdir(self.path)
        
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        image_path = 'Image/Test/Images/'
        mask_path = 'Mask/Test/Images/'
        image = Image.open(image_path+self.list[index])
        image = image.convert('RGB')
        mask = Image.open(mask_path+self.list[index])
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (image, mask)

    def __len__(self):
        return len(self.list)