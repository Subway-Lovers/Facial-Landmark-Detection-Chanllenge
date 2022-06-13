#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao
#
######################################################

import torch
import torchvision
import torch.nn as nn
import math


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14] [16, 48, 48]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7] [32, 24, 24]
        self.conv8 = nn.Conv2d(32, 128, 24, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(48) #[14] [48]
        self.avg_pool2 = nn.AvgPool2d(24) #[7] [24]
        self.fc = nn.Linear(176, 136) #196

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56] [64, 192, 192]
        
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56] [64, 192, 192]
        
        x = self.conv3_1(x)   #[64, 28, 28][64, 96, 96]
        
        x = self.block3_2(x)  #[64, 96, 96]
        
        x = self.block3_3(x)    #[64, 96, 96]
        
        x = self.block3_4(x)    #[64, 96, 96]
        
        out1 = self.block3_5(x) #[64, 96, 96]
        x = self.conv4_1(out1)  #[128, 48, 48]
        
        x = self.conv5_1(x)     #[128, 48, 48]
        
        x = self.block5_2(x)    #[128, 48, 48]
       
        x = self.block5_3(x)    #[128, 48, 48]
       
        x = self.block5_4(x)    #[128, 48, 48]
       
        x = self.block5_5(x)    #[128, 48, 48]
       
        x = self.block5_6(x)    #[128, 48, 48]
        
        x = self.conv6_1(x)     #[16, 14, 14] [16, 48, 48]
       
        x1 = self.avg_pool1(x)  #[16, 3, 3] [16, 1, 1]
        
        x1 = x1.view(x1.size(0), -1)    #[16]
        
        x = self.conv7(x)   #[32, 24, 24]
        
        x2 = self.avg_pool2(x)  #[32, 1, 1]
    
        x2 = x2.view(x2.size(0), -1)    #[32]
        
     
        x3 = self.relu(self.conv8(x)) #[128, 1, 1]
        
        x3 = x3.view(x3.size(0), -1) 
        #print('x3', x3.shape)
        

        multi_scale = torch.cat([x1, x2, x3], 1)
        #print(x1.shape, x2.shape, x3.shape, multi_scale.shape)
        landmarks = self.fc(multi_scale)

        return out1, landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(24, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 24, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        #[64, 28, 28] [64, 96, 96] [40, 96, 96][48, 24, 24]
        # print(x.shape)
        x = self.conv1(x) #[128, 14, 14] [128, 48, 48]
        
        x = self.conv2(x) #[128, 14, 14] [128, 48, 48]
       
        x = self.conv3(x) #[32, 7, 7] [32, 24, 24]
        
        x = self.conv4(x) #[128, 3, 3] [128, 3, 3]
        
        x = self.max_pool1(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

# if __name__ == '__main__':
#     input = torch.randn(1, 3, 112, 112)
#     pfld_backbone = PFLDInference()
#     auxiliarynet = AuxiliaryNet()
#     features, landmarks = pfld_backbone(input)
#     angle = auxiliarynet(features)

#     print("angle.shape:{0:}, landmarks.shape: {1:}".format(
#         angle.shape, landmarks.shape))
