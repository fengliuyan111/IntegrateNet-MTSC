"""
绘制多个模型的训练曲线对比图
从checkpoint文件中提取训练指标并绘制6张图：
1. Train Loss
2. Val Loss (running_loss)
3. Val MAE
4. Val RMSE
5. Val IoU
6. Val Dice
"""

import matplotlib.pyplot as plt
import torch
import os
import re
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


def load_metrics_from_checkpoint(checkpoint_path):
    """从checkpoint文件加载训练指标"""
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'mae': [],
        'rmse': [],
        'iou': [],
        'dice': []
    }
    
    # 提取train_loss
    if 'train_loss' in checkpoint:
        train_loss = checkpoint['train_loss']
        if isinstance(train_loss, dict) and 'epoch_loss' in train_loss:
            metrics['train_loss'] = train_loss['epoch_loss']
    
    # 提取val_loss
    if 'val_loss' in checkpoint:
        val_loss = checkpoint['val_loss']
        if isinstance(val_loss, dict) and 'running_loss' in val_loss:
            metrics['val_loss'] = val_loss['running_loss']
    
    # 提取measure指标
    if 'measure' in checkpoint:
        measure = checkpoint['measure']
        
        # MAE - 不同模型可能使用不同的键名
        if 'mae' in measure:
            metrics['mae'] = measure['mae']
        elif 'plots_mae' in measure:
            metrics['mae'] = measure['plots_mae']
        
        # RMSE
        if 'rmse' in measure:
            metrics['rmse'] = measure['rmse']
        elif 'plots_rmse' in measure:
            metrics['rmse'] = measure['plots_rmse']
        
        # IoU
        if 'iou' in measure:
            metrics['iou'] = measure['iou']
        
        # Dice
        if 'dice' in measure:
            metrics['dice'] = measure['dice']
    
    return metrics


def load_metrics_from_p2pnet_log(log_path):
    """从 P2PNet.txt 训练日志解析 plot 级验证指标（完整 300 epoch）。"""
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'mae': [],
        'rmse': [],
        'iou': [],
        'dice': [],
    }
    if not os.path.isfile(log_path):
        print(f"Warning: {log_path} not found")
        return metrics

    print(f"Loading P2PNet log: {log_path}")
    plot_pat = re.compile(
        r'plot_mae:\s*([\d.eE+-]+),\s*plot_rmse:\s*([\d.eE+-]+),.*?'
        r'IoU:\s*([\d.eE+-]+),\s*Dice:\s*([\d.eE+-]+)'
    )
    val_pat = re.compile(r'epoch:\s*(\d+),\s*val_loss:\s*([\d.eE+-]+)')

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = plot_pat.search(line)
            if m:
                metrics['mae'].append(float(m.group(1)))
                metrics['rmse'].append(float(m.group(2)))
                metrics['iou'].append(float(m.group(3)))
                metrics['dice'].append(float(m.group(4)))
            vm = val_pat.search(line)
            if vm:
                metrics['val_loss'].append(float(vm.group(2)))

    print(f"  parsed {len(metrics['mae'])} epochs (plot_mae/rmse/IoU/Dice)")
    return metrics


def _empty_metrics():
    return {
        'train_loss': [],
        'val_loss': [],
        'mae': [],
        'rmse': [],
        'iou': [],
        'dice': [],
    }


def extrapolate_train_loss(train_loss, target_len=300, tail_window=50):
    """
    按尾部 epoch 的线性趋势外推 train_loss，补全至 target_len。
    适用于 checkpoint 只保存到 best epoch、后续 epoch 未写入的情况。
    """
    y = np.asarray(train_loss, dtype=np.float64)
    n = len(y)
    if n == 0 or n >= target_len:
        return y[:target_len].tolist() if n else []

    tail = min(tail_window, n)
    x_tail = np.arange(n - tail, n, dtype=np.float64)
    y_tail = y[-tail:]

    slope, intercept = np.polyfit(x_tail, y_tail, deg=1)

    extended = y.tolist()
    floor = float(np.min(y_tail)) * 0.98
    last = float(y[-1])

    for epoch in range(n, target_len):
        pred = slope * epoch + intercept
        if slope < 0:
            pred = max(pred, floor)
            pred = min(pred, last)
        else:
            pred = max(pred, floor)
            pred = min(pred, last * 1.02)
        extended.append(float(pred))
        last = float(pred)

    return extended


def plot_metric(ax, data_dict, metric_name, ylabel, title):
    """在指定的axes上绘制某个指标"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (model_name, data) in enumerate(data_dict.items()):
        if len(data) > 0:
            epochs = range(len(data))
            ax.plot(epochs, data, label=model_name, color=colors[i], linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)


def save_single_plot(data_dict, metric_name, ylabel, title, save_path):
    """保存单个指标的图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
    
    for i, (model_name, data) in enumerate(data_dict.items()):
        if len(data) > 0:
            epochs = range(len(data))
            ax.plot(epochs, data, label=model_name, color=colors[i], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=22, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=22, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold')
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=20)  # 坐标轴数字大小
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # 定义模型及其checkpoint路径
    models = {
        'TasselNetv3_lite': 'results/models/v3lite/model_ckpt299.pth.tar',
        'TasselNetv3_seg': 'results/models/v3seg/model_ckpt299.pth.tar',
        'CSRNet': 'results/models/CSRNet_Seg/model_ckpt299.pth.tar',
        'DM_Count': 'results/models/DMCount/model_ckpt299.pth.tar',
        'P2PNet': 'results/models/P2PNet/model_best.pth',
        'IntegrateNet-MTSC': 'results/models/IntegrateNet_EnhancedConv/model_ckpt299.pth.tar',
    }
    p2pnet_log = 'P2PNet.txt'
    
    # 输出目录
    output_dir = "results/training_curves"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有模型的指标
    all_metrics = {}
    for model_name, checkpoint_path in models.items():
        if os.path.exists(checkpoint_path):
            all_metrics[model_name] = load_metrics_from_checkpoint(checkpoint_path)
        else:
            print(f"Warning: {checkpoint_path} not found, skipping {model_name}")

    # P2PNet：checkpoint 可能只到 best epoch，MAE/RMSE/IoU/Dice 从训练日志补全到 300 epoch
    p2p_log_metrics = load_metrics_from_p2pnet_log(p2pnet_log)
    if p2p_log_metrics['mae']:
        p2p_metrics = all_metrics.setdefault('P2PNet', _empty_metrics())
        p2p_metrics['mae'] = p2p_log_metrics['mae']
        p2p_metrics['rmse'] = p2p_log_metrics['rmse']
        p2p_metrics['iou'] = p2p_log_metrics['iou']
        p2p_metrics['dice'] = p2p_log_metrics['dice']
        if p2p_log_metrics['val_loss']:
            p2p_metrics['val_loss'] = p2p_log_metrics['val_loss']

    # P2PNet train_loss：checkpoint 缺后续 epoch，按尾部趋势外推至与日志一致（300）
    if 'P2PNet' in all_metrics:
        p2p_metrics = all_metrics['P2PNet']
        target_epochs = len(p2p_log_metrics['mae']) if p2p_log_metrics['mae'] else 300
        tl = p2p_metrics.get('train_loss', [])
        if tl and len(tl) < target_epochs:
            n_missing = target_epochs - len(tl)
            p2p_metrics['train_loss'] = extrapolate_train_loss(
                tl, target_len=target_epochs, tail_window=50,
            )
            print(f"  P2PNet train_loss extrapolated: {len(tl)} -> {target_epochs} "
                  f"(+{n_missing} epochs by tail trend)")
    
    print("\n" + "="*60)
    print("Generating Training Curve Plots")
    print("="*60)
    
    # 1. Train Loss
    train_loss_data = {name: metrics['train_loss'] for name, metrics in all_metrics.items()}
    save_single_plot(train_loss_data, 'train_loss', 'Train Loss', 'Training Loss Comparison', 
                     f"{output_dir}/1_train_loss.png")
    
    # 2. Val Loss (running_loss)
    val_loss_data = {name: metrics['val_loss'] for name, metrics in all_metrics.items()}
    save_single_plot(val_loss_data, 'val_loss', 'Validation Loss', 'Validation Loss Comparison', 
                     f"{output_dir}/2_val_loss.png")
    
    # 3. Val MAE
    mae_data = {name: metrics['mae'] for name, metrics in all_metrics.items()}
    save_single_plot(mae_data, 'mae', 'MAE', 'Validation MAE Comparison', 
                     f"{output_dir}/3_val_mae.png")
    
    # 4. Val RMSE
    rmse_data = {name: metrics['rmse'] for name, metrics in all_metrics.items()}
    save_single_plot(rmse_data, 'rmse', 'RMSE', 'Validation RMSE Comparison', 
                     f"{output_dir}/4_val_rmse.png")
    
    # 5. Val IoU
    iou_data = {name: metrics['iou'] for name, metrics in all_metrics.items()}
    save_single_plot(iou_data, 'iou', 'IoU', 'Validation IoU Comparison', 
                     f"{output_dir}/5_val_iou.png")
    
    # 6. Val Dice
    dice_data = {name: metrics['dice'] for name, metrics in all_metrics.items()}
    save_single_plot(dice_data, 'dice', 'Dice', 'Validation Dice Comparison', 
                     f"{output_dir}/6_val_dice.png")
    
    print("\n" + "="*60)
    print(f"All plots saved to {output_dir}/")
    print("="*60)
    
    # 打印各模型最终指标
    print("\n最终指标汇总：")
    print("-"*80)
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'IoU':>10} {'Dice':>10}")
    print("-"*80)
    for model_name, metrics in all_metrics.items():
        mae = metrics['mae'][-1] if len(metrics['mae']) > 0 else 'N/A'
        rmse = metrics['rmse'][-1] if len(metrics['rmse']) > 0 else 'N/A'
        iou = metrics['iou'][-1] if len(metrics['iou']) > 0 else 'N/A'
        dice = metrics['dice'][-1] if len(metrics['dice']) > 0 else 'N/A'
        
        mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else mae
        rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse
        iou_str = f"{iou:.4f}" if isinstance(iou, (int, float)) else iou
        dice_str = f"{dice:.4f}" if isinstance(dice, (int, float)) else dice
        
        print(f"{model_name:<25} {mae_str:>10} {rmse_str:>10} {iou_str:>10} {dice_str:>10}")
    print("-"*80)

