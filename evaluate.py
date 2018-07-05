import torch
from build_model import FCN_GCN
from torch.utils.data import DataLoader
from data_loader_EVAL import LungSegTest
from torchvision import transforms
import torch.nn.functional as F    
import numpy as np

from skimage import morphology, color, io, exposure

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def Inv_Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = np.logical_not(y_true.flatten())
    y_pred_f = np.logical_not(y_pred.flatten())
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true_f.sum() + y_pred_f.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape[:2]
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    
    img_hsv = color.rgb2hsv(img)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    # Load test data
    img_size = (1024, 1024)

    n_test = 61
    inp_shape = (1024,1024,3)
    batch_size=1

    # Load model
    net = FCN_GCN(1)

    net.load_state_dict(torch.load('Weights_221_2/cp_19_0.1336055189371109.pth'))
    net.eval()


    ious = np.zeros(n_test)
    dices = np.zeros(n_test)
    inv_dices = np.zeros(n_test)    
    seed = 1
    transformations_test = transforms.Compose([transforms.Resize(img_size),
                                      transforms.ToTensor()])  
    test_set = LungSegTest(transforms = transformations_test) 
    test_loader = DataLoader(test_set, batch_size=batch_size)
    

    i = 0
    for xx, yy, name in test_loader:
        #img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = net(xx)
        pred = F.sigmoid(pred)
        pred = pred.detach().numpy()[0,0,:,:]
        mask = yy.numpy()[0,0,:,:]
        xx = xx.numpy()[0,:,:,:].transpose(1,2,0)
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))

        # Binarize masks
        gt = mask > 0.5
        pr = pred > 0.5

        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr, 0.02 * np.prod(img_size))

        io.imsave('results/{}.png'.format(name[0][:-4]), masked(img, gt, pr, 1))

        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        inv_dices[i] = Inv_Dice(gt, pr) 

        i += 1
        if i == n_test:
            break

    print ('Mean IoU:', ious.mean())
    print ('Mean Dice:', dices.mean())
    print ('Mean Inv. Dice:', inv_dices.mean())
    
    
    
