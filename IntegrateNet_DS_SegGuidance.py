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


# ==================== 基础双卷积模块 ====================
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


# ==================== Seg-Logits引导模块 ====================
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


# ==================== IntegrateNet with DS + Seg-Logits Guidance ====================
class IntegrateNet(nn.Module):
    """
    改进策略：深度监督 + Seg-Logits引导
    
    核心思路：
    1. 基础DoubleConv（无残差、无CBAM）
    2. Deep Supervision（1/16和1/8尺度辅助监督）
    3. Seg-Logits Guidance（用分割信息引导计数特征）
    
    优势：
    - 保持简单的网络结构
    - 深度监督改善梯度流动
    - Seg-Logits引导增强空间定位能力
    """
    def __init__(self, use_deep_supervision=True, use_seg_guidance=True):
        super(IntegrateNet, self).__init__()
        self.use_deep_supervision = use_deep_supervision
        self.use_seg_guidance = use_seg_guidance
        
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
        
        # 基础双卷积模块（无增强）
        self.Doubleconv1 = DoubleConv(in_channels=128, out_channels=128)
        self.Doubleconv2 = DoubleConv(in_channels=64, out_channels=64)
        self.Doubleconv3 = DoubleConv(in_channels=64, out_channels=64)
        self.Doubleconv5 = DoubleConv(in_channels=264, out_channels=256)
        
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
        
        # Seg-Logits引导模块
        if self.use_seg_guidance:
            self.seg_guidance = SegLogitsGuidanceModule(channels=64)
        
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
        output = self.Doubleconv3(output)  # [B, 64, H, W] 共享特征
        
        # Seg-Logits引导
        if self.use_seg_guidance:
            # 先从共享特征预测seg_logits
            seg_logits = self.seg_head(output)  # [B, 1, H, W]
            
            # 使用seg_logits引导计数特征
            guided_count_feat, spatial_mask = self.seg_guidance(output, seg_logits)
            
            # 基于引导后的特征计算密度图和局部计数
            local_count = self.Counting(guided_count_feat)
            density = self.conv3(guided_count_feat)
            
            result = {
                'density': density,
                'local_count': local_count,
                'segmentation': seg_logits,
                'spatial_mask': spatial_mask  # 返回空间引导mask用于可视化
            }
        else:
            # 无引导版本
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
    print("IntegrateNet - DS + SegGuidance")
    print("="*60)
    
    # 测试训练模式
    net = IntegrateNet(use_deep_supervision=True, use_seg_guidance=True).cuda()
    net.train()
    x = torch.randn(2, 3, 256, 256).cuda()
    output_train = net(x)
    
    print("Training mode:")
    print(f"  Main density shape: {output_train['density'].shape}")
    print(f"  Aux density 16 shape: {output_train['aux_density_16'].shape}")
    print(f"  Aux density 8 shape: {output_train['aux_density_8'].shape}")
    print(f"  Local count shape: {output_train['local_count'].shape}")
    print(f"  Segmentation shape: {output_train['segmentation'].shape}")
    print(f"  Spatial mask shape: {output_train['spatial_mask'].shape}")
    
    # 测试推理模式
    net.eval()
    output_eval = net(x)
    print("\nEval mode:")
    print(f"  Density shape: {output_eval['density'].shape}")
    print(f"  Local count shape: {output_eval['local_count'].shape}")
    print(f"  Segmentation shape: {output_eval['segmentation'].shape}")
    print(f"  Spatial mask shape: {output_eval['spatial_mask'].shape}")
    print(f"  Has aux outputs: {'aux_density_16' in output_eval}")
    
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params/1e6:.2f}M")
    print("="*60)
    
    print("\n改进策略组合:")
    print("1. ✓ 基础 DoubleConv (无残差、无CBAM)")
    print("2. ✓ Deep Supervision (1/16 + 1/8尺度)")
    print("3. ✓ Seg-Logits Guidance (空间引导)")
    print("4. ✓ 训练时多监督 + 引导")
    print("5. ✓ 测试时无开销（辅助头不参与推理）")
    print("="*60)

