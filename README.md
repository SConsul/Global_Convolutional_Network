# Global Convolutional Network

GCN Architecture is proposed in the paper "Large Kernel Matters ——
Improve Semantic Segmentation by Global Convolutional Network"[[pdf]](https://arxiv.org/pdf/1703.02719.pdf)

The Global Convolution Network or GCN is an architecture proposed for the task of segmenting images. A segmenter has to perform 2 tasks: classification as well as localization. This has an inherent challenge as both tasks have inherent diametrically opposite demands.
While a classifier has to be tranformation and rotation invariant, a localizer has to sensitive to the same. The GCN architecture finds a balance of the two demands with the following properties:

    i. To retain spatial information, no fully connected layers are used and a FCN framework is adopted
    
    ii.For beter classification, a large kernel size is adopted to enable dense connections in feature maps
    
 ## Architecture
 To limit the number of parameters, while maintaining a large kernel, large symmetric separable filters are used.
  ### GCN Block
  The GCN Block is essentially a kx1 followed by 1xk convolution summed with a parallely computed 1xk followed by kx1 convolution. NOTE: the blocks are acting on feature maps and so channel width is larger than 3
  ### Boundary Refinement Block 
  The BR block iproves the segemetation near the boundaries of objects, where segmentation is less like a pure classification problem. It's design is inspired by that of ResNets and is basically a parallel brach of Conv+ReLu, folowed by another conv layer added to the input.
 
 ## Training 
   A pretrained ResNet-50 is used and is later fine-tuned. The rationale being that while medical images are vastly different from natural images, the ResNet is a good feature extractor (eg. edges, blobs, etc.) It is further augmented by the fact that many components in a medical image have features that resemble that of natural images eg. nucleui looks similar to balls.
 
Refer to http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006 for the ResNet50 Architecture and 

https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py for the torchvision.model code

### Loss Function
 
A linear combination of Soft Dice Loss, Soft Inverse Dice Loss, and Binary Cross-Entropy Loss (with logits) is used to train the model end-to-end. The best performance was obtained by weighing the three criteria at 50:25:25 (respectively)

#### Binary Cross-Entropy Loss (with logits)

This is calculated by passing the output of the network through a sigmoid activation before applying cross-entropy loss.

> The sigmoid and cross entropy calculations  are done in one class to exploit the log-sum-exp trick for greater numerical stability (as compared to 
sequentially applying sigmoid activation and then using vanilla BCE).

<!---$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top$ --->

$$l_n = - w_n \left[ t_n \cdot \log \sigma(x_n) + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right],$$

$$ L(x,y) = \sum_{i=1}^{N}l_i$$

#### Soft Dice Loss
 
Dice Loss gives a measure of how accurate the overlap of the mask and ground truth is.
The Sørensen–Dice coefficient is calculated as: $\frac{2. X\cap Y}{|X| + |Y|} = \frac{2. TP}{2. TP + FP + FN}$  and the Dice Loss is simply 1 - Dice coeff.
For Soft version of the loss, the output of the network is passed through a sigmoid before standard dice loss is evaluated.

 #### Soft Inverse Dice Loss

Inverse Dice loss checks for how accurately the background is masked. This penalizes the excess areas in the predicted mask. It is found by inverting the output before using the soft dice loss. 
