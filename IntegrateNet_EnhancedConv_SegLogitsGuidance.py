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


# ==================== 改进1: CBAM注意力机制 ====================
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


# ==================== 改进1: 增强的双卷积模块 ====================
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


# ==================== 基础模块 ====================
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


# ==================== 改进2: 分割Logits引导模块 ====================
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


# ==================== IntegrateNet with EnhancedConv + SegLogitsGuidance ====================
class IntegrateNet(nn.Module):
    """
    综合两种改进:
    1. 增强的双卷积: 残差连接 + CBAM注意力机制
    2. 分割Logits引导: 使用分割logits引导计数特征
    """
    def __init__(self, use_seg_guidance=True):
        super(IntegrateNet, self).__init__()
        self.use_seg_guidance = use_seg_guidance
        
        # 原始MixNet编码器
        self.Encoder = MixNet()
        
        # CARAFE上采样
        self.CARAFE1 = CARAFE_upsampling(256,128)
        self.CARAFE2 = CARAFE_upsampling(128,64)
        self.CARAFE3 = CARAFE_upsampling(64,64,delta=8)
        
        # 通道调整
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(1,1), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=56, out_channels=64, kernel_size=(1,1), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=(1,1))
        
        # 改进1: 增强的双卷积（与已训练权重一致：跳跃连接用通道拼接）
        self.Doubleconv1 = EnhancedDoubleConv(in_channels=256, out_channels=128, use_attention=True)
        self.Doubleconv2 = EnhancedDoubleConv(in_channels=128, out_channels=64, use_attention=True)
        self.Doubleconv3 = EnhancedDoubleConv(in_channels=64, out_channels=64, use_attention=True)
        self.Doubleconv5 = EnhancedDoubleConv(in_channels=264, out_channels=256, use_attention=True)
        
        # 分割头：从共享特征直接预测
        self.seg_head = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        # 改进2: 分割Logits引导模块
        if self.use_seg_guidance:
            self.seg_guidance = SegLogitsGuidanceModule(channels=64)
        
        self.Counting = Counter()
        self.weights_init()

    def forward(self, output):
        imgsize = output.size()
        dic = self.Encoder(output)
        one_8, one_16, one_32 = dic['one_8'], dic['one_16'], dic['one_32']
        
        # 上采样阶段1: 1/32 -> 1/16
        output = self.Doubleconv5(one_32)
        output = self.CARAFE1(output)
        one_16 = self.conv1(one_16)
        if output.size() != one_16.size():
            output = TF.resize(output,size=one_16.size()[2:])
        output = torch.cat((one_16, output), dim=1)
        output = self.Doubleconv1(output)

        # 上采样阶段2: 1/16 -> 1/8
        output = self.CARAFE2(output)
        one_8 = self.conv2(one_8)
        if output.size() != one_8.size():
            output = TF.resize(output,size=one_8.size()[2:])
        output = torch.cat((one_8, output), dim=1)
        output = self.Doubleconv2(output)

        # 上采样阶段3: 1/8 -> 1/1
        output = self.CARAFE3(output)
        if output.size()[2:] != imgsize[2:]:
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
    print("IntegrateNet - EnhancedConv + SegLogitsGuidance")
    print("="*60)
    
    # Test with both improvements
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
    
    print("\n改进要点:")
    print("1. ✓ 增强的双卷积模块 (残差连接 + CBAM注意力)")
    print("2. ✓ 分割Logits引导机制 (用分割logits直接引导计数)")
    print("="*60)

