# Global_Convolutional_Network
Pytorch implementation of GCN architecture for sematic segmentation

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
  ### Boundarty Refinement Block 
  The BR block iproves the segemetation near the boundaries of objects, where segmentation is less like a pure classification problem. It's design is inspired by that of ResNets and is basically a parallel brach of Conv+ReLu, folowed by another conv layer added to the input.
 
 ## Training 
   A pretrained ResNet-50 is used and is later fine-tuned. This works as medical images have features that resemble that of other natural images eg. nucleus looks similar to a ball.
 
    Refer to http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006 for the ResNet50 Architecture and 

    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py for the torchvision.model code
