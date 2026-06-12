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


# ==================== IntegrateNet with Deep Supervision ====================
class IntegrateNet(nn.Module):
    """
    改进策略：深度监督 (Deep Supervision)
    
    核心思路：
    1. 保持EnhancedDoubleConv（已验证有效）
    2. 在中间层(1/16和1/8)添加辅助密度预测头
    3. 训练时多个监督信号改善梯度流动
    4. 测试时只使用主输出（无额外计算开销）
    
    优势：
    - 不改变网络架构的特征提取能力
    - 改善深层网络的梯度消失问题
    - 辅助损失约束中间特征学习更好的表示
    - 已被UNet++, HRNet等论文验证有效
    """
    def __init__(self, use_deep_supervision=True):
        super(IntegrateNet, self).__init__()
        self.use_deep_supervision = use_deep_supervision
        
        # MixNet编码器
        self.Encoder = MixNet()
        
        # CARAFE上采样
        self.CARAFE1 = CARAFE_upsampling(256,128)
        self.CARAFE2 = CARAFE_upsampling(128,64)
        self.CARAFE3 = CARAFE_upsampling(64,64,delta=8)
        
        # 通道调整
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=(1,1), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=56, out_channels=64, kernel_size=(1,1), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=(1,1))
        
        # 增强的双卷积模块（保持不变）
        self.Doubleconv1 = EnhancedDoubleConv(in_channels=128, out_channels=128, use_attention=True)
        self.Doubleconv2 = EnhancedDoubleConv(in_channels=64, out_channels=64, use_attention=True)
        self.Doubleconv3 = EnhancedDoubleConv(in_channels=64, out_channels=64, use_attention=True)
        self.Doubleconv5 = EnhancedDoubleConv(in_channels=264, out_channels=256, use_attention=True)
        
        # 深度监督：辅助密度预测头
        if self.use_deep_supervision:
            # 1/16尺度的辅助头
            self.aux_head_16 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1)
            )
            # 1/8尺度的辅助头
            self.aux_head_8 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1)
            )
        
        # 主任务头
        self.Counting = Counter()
        self.seg_head = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        self.weights_init()

    def forward(self, output):
        imgsize = output.size()
        dic = self.Encoder(output)
        one_8, one_16, one_32 = dic['one_8'], dic['one_16'], dic['one_32']
        
        # Stage 1: 1/32 -> 1/16
        output = self.Doubleconv5(one_32)
        output = self.CARAFE1(output)
        one_16 = self.conv1(one_16)
        if output.size() != one_16.size():
            output = TF.resize(output, size=one_16.size()[2:])
        output = one_16 + output
        output_16 = self.Doubleconv1(output)  # 保存1/16特征
        
        # 辅助监督1：1/16尺度密度图
        if self.training and self.use_deep_supervision:
            aux_density_16 = self.aux_head_16(output_16)
            # 上采样到原始尺寸
            aux_density_16 = F.interpolate(aux_density_16, size=imgsize[2:], 
                                          mode='bilinear', align_corners=False)
        
        # Stage 2: 1/16 -> 1/8
        output = self.CARAFE2(output_16)
        one_8 = self.conv2(one_8)
        if output.size() != one_8.size():
            output = TF.resize(output, size=one_8.size()[2:])
        output = one_8 + output
        output_8 = self.Doubleconv2(output)  # 保存1/8特征
        
        # 辅助监督2：1/8尺度密度图
        if self.training and self.use_deep_supervision:
            aux_density_8 = self.aux_head_8(output_8)
            # 上采样到原始尺寸
            aux_density_8 = F.interpolate(aux_density_8, size=imgsize[2:], 
                                         mode='bilinear', align_corners=False)
        
        # Stage 3: 1/8 -> 1/1
        output = self.CARAFE3(output_8)
        if output.size()[2:] != imgsize[2:]:
            output = TF.resize(output, size=imgsize[2:])
        output = self.Doubleconv3(output)
        
        # 主输出
        seg_logits = self.seg_head(output)
        local_count = self.Counting(output)
        density = self.conv3(output)
        
        result = {
            'density': density,
            'local_count': local_count,
            'segmentation': seg_logits
        }
        
        # 返回辅助输出（仅训练时）
        if self.training and self.use_deep_supervision:
            result['aux_density_16'] = aux_density_16
            result['aux_density_8'] = aux_density_8
        
        return result

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
    print("IntegrateNet - EnhancedConv + Deep Supervision")
    print("="*60)
    
    # 测试训练模式
    net = IntegrateNet(use_deep_supervision=True).cuda()
    net.train()
    x = torch.randn(2, 3, 256, 256).cuda()
    output_train = net(x)
    
    print("Training mode:")
    print(f"  Main density shape: {output_train['density'].shape}")
    print(f"  Aux density 16 shape: {output_train['aux_density_16'].shape}")
    print(f"  Aux density 8 shape: {output_train['aux_density_8'].shape}")
    print(f"  Local count shape: {output_train['local_count'].shape}")
    print(f"  Segmentation shape: {output_train['segmentation'].shape}")
    
    # 测试推理模式
    net.eval()
    output_eval = net(x)
    print("\nEval mode:")
    print(f"  Density shape: {output_eval['density'].shape}")
    print(f"  Local count shape: {output_eval['local_count'].shape}")
    print(f"  Segmentation shape: {output_eval['segmentation'].shape}")
    print(f"  Has aux outputs: {'aux_density_16' in output_eval}")
    
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params/1e6:.2f}M")
    print("="*60)
    
    print("\n改进策略 - 深度监督:")
    print("1. ✓ EnhancedDoubleConv (残差 + CBAM) - 保持不变")
    print("2. ✓ 1/16尺度辅助监督 - 改善中层特征学习")
    print("3. ✓ 1/8尺度辅助监督 - 改善浅层特征学习") 
    print("4. ✓ 训练时多监督 - 改善梯度流动")
    print("5. ✓ 测试时无开销 - 只用主输出")
    print("="*60)

