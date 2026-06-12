"""
Grad-CAM热力图可视化脚本
可视化 IntegrateNet_EnhancedConv 模型在不同解码器层关注的区域
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from Netdataset import BlueberryDataset, BBNormalize, BBToTensor, BBZeroPadding
from IntegrateNet import IntegrateNet


class GradCAM:
    """Grad-CAM实现：用于可视化CNN关注的区域"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        handle_f = self.target_layer.register_forward_hook(forward_hook)
        handle_b = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([handle_f, handle_b])
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def generate_cam(self, input_tensor, target_output='density'):
        """生成Grad-CAM热力图"""
        self.model.eval()
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 获取目标输出
        if target_output == 'density':
            target = output['density'].sum()
        elif target_output == 'segmentation':
            target = output['segmentation'].sum()
        else:
            target = output['density'].sum()
        
        # 反向传播
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # 计算Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, output


def save_original_image(image, save_path):
    """保存原始图像"""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {save_path}")


def save_gradcam_heatmap(image, cam, save_path):
    """保存Grad-CAM热力图（叠加在原图上）"""
    plt.figure(figsize=(10, 8))
    
    # 上采样到原始图像尺寸
    cam_resized = cv.resize(cam, (image.shape[1], image.shape[0]))
    
    plt.imshow(image)
    plt.imshow(cam_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # 配置
    data_dir = "dataset_blueberry_augmented"
    test_list = "final_test.txt"
    
    image_scale = 1. / 255
    image_mean = np.array([0.3751, 0.4294, 0.3128]).reshape((1, 1, 3))
    image_std = np.array([1, 1, 1]).reshape((1, 1, 3))
    output_stride = 8
    
    # 输出目录
    output_dir = "results/gradcam_visualization_Integrate"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    val_transforms = transforms.Compose([
        BBNormalize(scale=image_scale, std=image_std, mean=image_mean, train=False),
        BBToTensor(train=False),
        BBZeroPadding(output_stride, train=False)
    ])
    
    valset = BlueberryDataset(
        data_dir=data_dir, 
        data_list=test_list,
        train=False, 
        transform=val_transforms
    )
    
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    
    # 加载IntegrateNet模型
    print("Loading IntegrateNet...")
    model_path = "results/models/IntegrateNet/model_best.pth"
    
    net = IntegrateNet()
    checkpoint = torch.load(model_path, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # 处理DataParallel的键名
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    
    net.load_state_dict(state_dict)
    net = net.cuda()
    net.eval()
    
    # 只可视化 1/8 scale (Doubleconv2)
    target_layer = net.Doubleconv2
    
    print("="*60)
    print("Generating Grad-CAM Heatmap (1/8 scale)")
    print("="*60)
    
    # 处理前5张图像
    for idx, sample in enumerate(val_loader):
        if idx >= 5:
            break
            
        input_image = sample['image'].cuda()
        filename = sample.get('filename', [f'image_{idx}'])[0]
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        print(f"\nProcessing {idx+1}/5: {base_name}")
        
        # 获取原始图像用于可视化
        img_np = input_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * image_std + image_mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        # 生成 1/8 scale 的 Grad-CAM
        grad_cam = GradCAM(net, target_layer)
        with torch.enable_grad():
            input_clone = input_image.clone().requires_grad_(True)
            cam, output = grad_cam.generate_cam(input_clone, 'density')
        grad_cam.remove_hooks()
        
        # 单独保存原始图像
        save_path_orig = f"{output_dir}/{base_name}_original.png"
        save_original_image(img_np, save_path_orig)
        
        # 单独保存Grad-CAM热力图
        save_path_cam = f"{output_dir}/{base_name}_gradcam.png"
        save_gradcam_heatmap(img_np, cam, save_path_cam)
    
    print("\n" + "="*60)
    print(f"All Grad-CAM visualizations saved to {output_dir}/")
    print("="*60)
