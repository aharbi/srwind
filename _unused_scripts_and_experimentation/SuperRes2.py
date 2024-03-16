import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# following basic setup via Pytorch tutorial, but focused on our UNet architecture for SR
#
# The "SuperRes" class is based on a UNet architecture.
# Input: single low-res image, upsampled via bicubic interpolation
# Output: network weights for accurate result

class SuperRes(nn.Module):
  
    # init function
    def __init__(self, numFeatures=256, convKernel=3, imDims=(100,100), numDownsampleBlocks=2, numMidConvBlocks=2):
        super(SuperRes, self).__init__()

        # init self props
        self.numFeatures = numFeatures
        self.convKernel = convKernel
        self.imDims = imDims
        self.numDownsampleBlocks = numDownsampleBlocks
        self.numMidConvBlocks = numMidConvBlocks

        # set up the layers
        # "in" -- labels set of funnel input down-conv and pooling layers
        # "mid" -- labels the middle functional layers
        # "out" -- labels up-conv layers

        # QUESTIONS -- 
        #  how to pick how many in/out channels?
        #  how to up-convolve?
        #  difference between nn.F relu and nn.relu? functional vs. layer?
        
        # in layers
        self.inBlockIncreaseFeatures = nn.Sequential(
            # block to increase features from 2 (input) to full specified number
            nn.Conv2d(in_channels=2, out_channels=self.numFeatures/2, kernel_size=self.convKernel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.numFeatures/2, out_channels=self.numFeatures, kernel_size=self.convKernel),
            nn.ReLU()
        )

        self.convBlockDS = nn.Sequential(
            nn.Conv2d(in_channels=self.numFeatures, out_channels=self.numFeatures, kernel_size=self.convKernel),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.numFeatures, out_channels=self.numFeatures, kernel_size=self.convKernel),
            nn.ReLU(),
        )

        self.inBlockOutDims = (self.imDims-self.convKernel+1, self.imDims-self.convKernel+1)
        self.avgPool  = nn.AvgPool2d(kernel_size=2) # assumes an odd convKernel, yielding an even dimension reduction        
        

        # out (up-scale image)
        self.outBlockUS = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.numFeatures, out_channels=self.numFeatures, kernel_size=self.convKernel),
            nn.Relu(),
            nn.ConvTranspose2d(in_channels=self.numFeatures, out_channels=self.numFeatures, kernel_size=self.convKernel),
            nn.Relu(), 
        )

        self.outBlockReduceFeatures = nn.Sequential(
            # block to reduce features from full num to 2 (output)
            nn.ConvTranspose2d(in_channels=self.numFeatures, out_channels=self.numFeatures/2, kernel_size=self.convKernel),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.numFeatures/2, out_channels=2, kernel_size=self.convKernel),
            nn.ReLU()
        )

    
    # NEED TO CONFIRM DIMENSIONALITIES OF DATA
    def forward(self, x):
        
        # INPUT LAYERS
        x = self.inBlockIncreaseFeatures(x)

        for ii in np.arange(self.numDownsampleBlocks):
            # 1: check current dimensionality of image
            # 2: pad if necessary
            
            # 3: get average pool
            x = self.avgPool(x) # cut size by pool

            # 4: take 2 convolution/relu blocks
            x = self.convBlockDS(x) # cut size by conv                

        # MID LAYERS - additional convolution blocks
        for ii in np.arange(self.numMidConvBlocks):
            x = self.convBlockDS(x)

        # OUT LAYERS
        for ii in np.arange(self.numDownsampleBlocks):
            x = self.outBlockUS(x)

        x = self.outBlockReduceFeatures(x)     

        # QUESTION: take softmax at end?   

        # RETURN
        return x

