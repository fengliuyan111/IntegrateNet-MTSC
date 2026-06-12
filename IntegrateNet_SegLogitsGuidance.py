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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Counter(nn.Module):
    def __init__(self):
        super(Counter, self).__init__()
        self.pool=nn.AvgPool2d(kernel_size=64,stride=64)
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),stride=(1,1))
        self.conv2=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1),stride=(1,1))
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(1)

    def forward(self,x):
        x=self.pool(x)
        x=F.relu(self.bn1(self.conv1(x)),inplace=True)
        x=F.relu(self.bn2(self.conv2(x)),inplace=True)
        return x


class SegLogitsGuidanceModule(nn.Module):
    """使用分割logits直接引导计数特征"""
    def __init__(self, channels=64):
        super(SegLogitsGuidanceModule, self).__init__()
        # 将seg_logits (1通道) 转换为引导mask
        self.mask_refine = nn.Sequential(
            nn.Conv2d(1, channels // 4, 3, 1, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()  # 输出0-1的引导权重
        )
        
        # 融合引导后的特征
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 可学习的融合权重
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, count_feat, seg_logits):
        """
        count_feat: [B, 64, H, W] 计数分支特征
        seg_logits: [B, 1, H, W] 分割logits（未经过sigmoid）
        """
        # 确保seg_logits和count_feat尺寸一致
        if seg_logits.size()[-2:] != count_feat.size()[-2:]:
            seg_logits = F.interpolate(seg_logits, size=count_feat.size()[-2:], 
                                      mode='bilinear', align_corners=True)
        
        # 从seg_logits生成精细的空间引导mask
        spatial_mask = self.mask_refine(seg_logits)
        
        # 使用mask引导计数特征（加权增强）
        guided_count_feat = count_feat * (1.0 + self.gamma * spatial_mask)
        
        # 融合特征
        output = self.fusion(guided_count_feat)
        
        return output, spatial_mask


class IntegrateNet(nn.Module):
    def __init__(self, use_seg_guidance=True):
        super(IntegrateNet, self).__init__()
        self.use_seg_guidance = use_seg_guidance
        
        self.Encoder = MixNet()
        self.CARAFE1 = CARAFE_upsampling(256,128)
        self.CARAFE2 = CARAFE_upsampling(128,64)
        self.CARAFE3 = CARAFE_upsampling(64,64,delta=8)
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=56, out_channels=64,kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=(1,1),dilation=(1,1),padding=(0,0))
        
        self.Doubleconv1 = DoubleConv(in_channels=128,out_channels=128)
        self.Doubleconv2 = DoubleConv(in_channels=64,out_channels=64)
        self.Doubleconv3 = DoubleConv(in_channels=64,out_channels=64)
        self.Doubleconv5 = DoubleConv(in_channels=264,out_channels=256)
        
        # 分割头：从共享特征直接预测
        self.seg_head = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        if self.use_seg_guidance:
            # 使用seg_logits引导的模块
            self.seg_guidance = SegLogitsGuidanceModule(channels=64)
        
        self.Counting = Counter()
        self.weights_init()

    def forward(self, output):
        imgsize = output.size()
        dic = self.Encoder(output)
        one_8, one_16, one_32 = dic['one_8'], dic['one_16'], dic['one_32']
        
        output = self.Doubleconv5(one_32)
        output = self.CARAFE1(output)
        one_16 = self.conv1(one_16)
        if output.size() != one_16.size():
            output = TF.resize(output,size=one_16.size()[2:])
        output = one_16 + output
        output = self.Doubleconv1(output)

        output = self.CARAFE2(output)
        one_8 = self.conv2(one_8)
        if output.size() != one_8.size():
            output = TF.resize(output,size=one_8.size()[2:])
        output = one_8 + output
        output = self.Doubleconv2(output)

        output = self.CARAFE3(output)
        if output != imgsize:
            output = TF.resize(output,size=imgsize[2:])
        output = self.Doubleconv3(output)  # [B, 64, H, W] 共享特征

        if self.use_seg_guidance:
            # 先从共享特征预测seg_logits
            seg_logits = self.seg_head(output)  # [B, 1, H, W]
            
            # 使用seg_logits引导计数特征
            guided_count_feat, spatial_mask = self.seg_guidance(output, seg_logits)
            
            # 基于引导后的特征计算密度图和局部计数
            local_count = self.Counting(guided_count_feat)
            density = self.conv3(guided_count_feat)
            
            return {
                'density': density, 
                'local_count': local_count, 
                'segmentation': seg_logits,
                'spatial_mask': spatial_mask  # 返回空间引导mask用于可视化
            }
        else:
            # 原始版本（无引导）
            seg_logits = self.seg_head(output)
            local_count = self.Counting(output)
            density = self.conv3(output)
            return {'density': density, 'local_count': local_count, 'segmentation': seg_logits}

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
    print("="*60)
    print("Testing IntegrateNet with Seg-Logits Guidance")
    print("="*60)
    
    # Test with seg-logits guidance
    net = IntegrateNet(use_seg_guidance=True).cuda()
    x = torch.randn(2, 3, 256, 256).cuda()
    out = net(x)
    
    print(f"Density shape: {out['density'].shape}")
    print(f"Local count shape: {out['local_count'].shape}")
    print(f"Segmentation shape: {out['segmentation'].shape}")
    print(f"Spatial mask shape: {out['spatial_mask'].shape}")
    
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print("="*60)
    
    # Test without guidance
    print("\nTesting without seg-logits guidance:")
    net_baseline = IntegrateNet(use_seg_guidance=False).cuda()
    out_baseline = net_baseline(x)
    print(f"Density shape: {out_baseline['density'].shape}")
    print(f"Local count shape: {out_baseline['local_count'].shape}")
    print(f"Segmentation shape: {out_baseline['segmentation'].shape}")
    baseline_params = sum(p.numel() for p in net_baseline.parameters() if p.requires_grad)
    print(f"Total parameters: {baseline_params/1e6:.2f}M")
    print("="*60)

