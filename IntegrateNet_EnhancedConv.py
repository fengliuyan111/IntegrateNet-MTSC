from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from sympy import *
import os
import json
import pandas as pd
import random
from PIL import Image
import h5py
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from skimage import util
from skimage.measure import label
from skimage.measure import regionprops
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from mixnetL import *
from CARAFE import *


# ==================== CBAM注意力机制 ====================
class CBAM(nn.Module):
    """Convolutional Block Attention Module - 用于增强特征表达"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


# ==================== 增强的双卷积模块 ====================
class EnhancedDoubleConv(nn.Module):
    """增强的双卷积模块 - 使用残差连接和注意力机制"""
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(EnhancedDoubleConv, self).__init__()
        self.use_attention = use_attention
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # 注意力机制
        if use_attention:
            self.attention = CBAM(out_channels)
        
        # 残差连接的通道调整
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv(x)
        if self.use_attention:
            out = self.attention(out)
        out = out + identity
        return F.relu(out, inplace=True)


class Counter(nn.Module):
    def __init__(self):
        super(Counter, self).__init__()
        self.pool=nn.AvgPool2d(kernel_size=64,stride=64)  # 使用stride=64，与ground truth保持一致
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),stride=(1,1))
        self.conv2=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1),stride=(1,1))
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(1)

    def forward(self,x):
        x=self.pool(x)  # 输出4x4大小
        x=F.relu(self.bn1(self.conv1(x)),inplace=True)
        x=F.relu(self.bn2(self.conv2(x)),inplace=True)
        return x


class IntegrateNet(nn.Module):
    def __init__(self):
        super(IntegrateNet, self).__init__()
        self.Encoder = MixNet()
        self.CARAFE1 = CARAFE_upsampling(256,128)
        self.CARAFE2 = CARAFE_upsampling(128,64)
        self.CARAFE3 = CARAFE_upsampling(64,64,delta=8)
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=56, out_channels=64,kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        
        # 使用增强的双卷积模块（带残差连接和注意力机制）
        self.Doubleconv1 = EnhancedDoubleConv(in_channels=128, out_channels=128, use_attention=True)
        self.Doubleconv2 = EnhancedDoubleConv(in_channels=64, out_channels=64, use_attention=True)
        self.Doubleconv3 = EnhancedDoubleConv(in_channels=64, out_channels=64, use_attention=True)
        self.Doubleconv5 = EnhancedDoubleConv(in_channels=264, out_channels=256, use_attention=True)
        
        self.Counting = Counter()
        # segmentation head branching from decoder output (keep original counting intact)
        self.seg_head = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.weights_init()

    def forward(self, output):
        # upsampling #1
        imgsize = output.size()
        dic = self.Encoder(output)
        one_8, one_16, one_32 = dic['one_8'], dic['one_16'], dic['one_32']
        output = self.Doubleconv5(one_32)
        output = self.CARAFE1(output)
        one_16 = self.conv1(one_16)
        if output.size() != one_16.size(): ## avoid different sizes through upsampling
            output = TF.resize(output,size=one_16.size()[2:])

        # 逐元素相加（论文实现）
        output = one_16 + output  # 保持128通道
        output = self.Doubleconv1(output)  # 128->128通道

        # upsampling #2
        output = self.CARAFE2(output)
        one_8 = self.conv2(one_8)
        if output.size() != one_8.size():
            output = TF.resize(output,size=one_8.size()[2:])

        # 逐元素相加（论文实现）
        output = one_8 + output  # 保持64通道
        output = self.Doubleconv2(output)  # 64->64通道

        # upsampling #3
        output = self.CARAFE3(output)
        if output != imgsize:
            output = TF.resize(output,size=imgsize[2:])
        output = self.Doubleconv3(output)

        # segmentation logits at full resolution
        seg_logits = self.seg_head(output)
        local_count = self.Counting(output)
        output = self.conv3(output)
        return {'density': output, 'local_count': local_count, 'segmentation': seg_logits}

    ## initial weights
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = IntegrateNet().cuda()
    x = torch.randn(2, 3, 256, 256).cuda()
    output = net(x)
    print("="*60)
    print("IntegrateNet with Enhanced DoubleConv (Residual + CBAM)")
    print("="*60)
    print(f"Density map shape: {output['density'].shape}")
    print(f"Local count shape: {output['local_count'].shape}")
    print(f"Segmentation shape: {output['segmentation'].shape}")
    
    # 统计参数量
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    print("="*60)

