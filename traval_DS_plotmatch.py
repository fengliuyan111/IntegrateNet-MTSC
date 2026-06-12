import os
# Set PyTorch memory allocator to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
from time import time
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import torch.optim as optim
from error import *
from IntegrateNet_DS import *
from Netdataset import *


# system-related parameters
data_dir = "dataset"
train_list = "train.txt"
val_list = "val.txt"

# normalization
image_scale = 1. / 255
image_mean = [0.3751, 0.4294, 0.3128]
image_std = [1, 1, 1]
image_mean = np.array(image_mean).reshape((1, 1, 3))
image_std = np.array(image_std).reshape((1, 1, 3))
crop_num = 4
input_size = 64
output_stride = 8

# model-related parameters
optimizer = 'sgd'
batch_size = 16
crop_size = (256,256)
learning_rate = 0.01
momentum = 0.95
mult = 1
num_epoch = 300
weight_decay = 0.0005
mae_max = 10000

# 深度监督权重
aux_weight_16 = 0.4  # 1/16尺度辅助损失权重
aux_weight_8 = 0.6   # 1/8尺度辅助损失权重（越接近输出权重越大）


def save_checkpoint(state, snapshot_dir, filename='model_ckpt.pth.tar'):
    os.makedirs(snapshot_dir, exist_ok=True)
    torch.save(state, '{}/{}'.format(snapshot_dir, filename))


def load_checkpoint(checkpoint_path, net, optimizer=None):
    """加载checkpoint并返回epoch和历史记录"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, None
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    net.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.state_dict()  # 先获取当前optimizer的state_dict
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('train_loss', {'running_loss': [], 'epoch_loss': []})
    val_loss = checkpoint.get('val_loss', {'running_loss': [], 'epoch_loss': []})
    measure = checkpoint.get('measure', {'plots_mae': [], 'plots_rmse': [], 'plots_r2': []})
    
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch, {'train_loss': train_loss, 'val_loss': val_loss, 'measure': measure}


# training part with deep supervision
def train(net, train_loader, optimizer, criterion, criterion2, epoch, lamda, seg_weight=0.0, bce_loss=None):
    net.train()
    target_filter = torch.ones((1, 1, 64, 64), dtype=torch.float32, device='cuda')
    running_loss = 0.0
    running_main_loss = 0.0
    running_aux_loss = 0.0

    for i, sample in enumerate(train_loader):
        inputs, targets = sample['image'], sample['target']
        masks = sample.get('mask', None)
        length = len(inputs)
        for j in range(length):
            input,target = inputs[j],targets[j]
            input,target = input.cuda(), target.cuda()
            mask = None
            if masks is not None:
                mask = masks[j].cuda()

            optimizer.zero_grad()
            dic = net(input)
            density = dic['density']
            local_count = dic['local_count']
            seg_logits = dic.get('segmentation', None)

            # 主损失
            loss1 = criterion(density, target)
            target_count = F.conv2d(target, target_filter, stride=64)
            loss2 = criterion2(local_count, target_count)
            main_loss = (1-lamda)*loss1+lamda*loss2
            
            # 深度监督损失
            aux_loss = 0
            if 'aux_density_16' in dic:
                aux_loss_16 = criterion(dic['aux_density_16'], target)
                aux_loss += aux_weight_16 * aux_loss_16
            if 'aux_density_8' in dic:
                aux_loss_8 = criterion(dic['aux_density_8'], target)
                aux_loss += aux_weight_8 * aux_loss_8
            
            # 分割损失
            seg_loss = 0
            if seg_weight > 0 and (seg_logits is not None) and (mask is not None) and (bce_loss is not None):
                if seg_logits.size()[-2:] != mask.size()[-2:]:
                    mask_ds = F.interpolate(mask, size=seg_logits.size()[-2:], mode='nearest')
                else:
                    mask_ds = mask
                seg_loss = bce_loss(seg_logits, mask_ds)
            
            # 总损失
            loss = main_loss + aux_loss + seg_weight * seg_loss

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_main_loss += main_loss.item()
            running_aux_loss += aux_loss if isinstance(aux_loss, float) else aux_loss.item()

    net.train_loss['epoch_loss'].append(running_loss / (length*(i + 1)))
    torch.cuda.empty_cache()


# testing part
def validate(net, val_loader, epoch,criterion,criterion2,lamda, seg_weight=0.0, bce_loss=None, eval_with_mask=False):
    net.eval()

    target_filter = torch.ones((1, 1, 64, 64), dtype=torch.float32, device='cuda')
    iou_list = []
    dice_list = []
    running_loss = 0.0
    all_pd_counts = []
    all_gt_counts = []
    
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image, target = sample['image'], sample['target']
            mask = sample.get('mask', None)
            pts = sample.get('points', None)
            input = image.cuda()
            target = target.cuda()

            _, _, h, w = input.shape
            max_size = 3200
            if h > max_size or w > max_size:
                stride = 512
                crop_size = 768
                output_full = torch.zeros((1, 1, h, w), device='cuda')
                seg_logits_full = torch.zeros((1, 1, h, w), device='cuda') if seg_weight > 0 else None
                count_map = torch.zeros((1, 1, h, w), device='cuda')
                
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        y_end = min(y + crop_size, h)
                        x_end = min(x + crop_size, w)
                        y_start = max(0, y_end - crop_size)
                        x_start = max(0, x_end - crop_size)
                        
                        patch = input[:, :, y_start:y_end, x_start:x_end]
                        dic_patch = net(patch)
                        out_patch = dic_patch['density']
                        seg_patch = dic_patch.get('segmentation', None)
                        
                        output_full[:, :, y_start:y_end, x_start:x_end] += out_patch
                        if seg_logits_full is not None and seg_patch is not None:
                            seg_logits_full[:, :, y_start:y_end, x_start:x_end] += seg_patch
                        count_map[:, :, y_start:y_end, x_start:x_end] += 1.0
                
                output = output_full / count_map.clamp(min=1.0)
                seg_logits = seg_logits_full / count_map.clamp(min=1.0) if seg_logits_full is not None else None
                local_count = None
            else:
                dic = net(input)
                output = dic['density']
                local_count = dic['local_count']
                seg_logits = dic.get('segmentation', None)

            loss1 = criterion(output, target)
            if local_count is not None:
                d = F.conv2d(target, target_filter, stride=64)
                loss2 = criterion2(local_count, d)
                loss = (1-lamda)*loss1+lamda*loss2
            else:
                loss = loss1
            
            if seg_weight > 0 and (seg_logits is not None) and (mask is not None) and (bce_loss is not None):
                mask = mask.cuda()
                if seg_logits.size()[-2:] != mask.size()[-2:]:
                    mask_ds = F.interpolate(mask, size=seg_logits.size()[-2:], mode='nearest')
                else:
                    mask_ds = mask
                loss_seg = bce_loss(seg_logits, mask_ds)
                loss = loss + seg_weight * loss_seg
            
            running_loss += loss.item()
            output = output.squeeze().cpu().detach().numpy()
            output = np.clip(output,0,None)

            if eval_with_mask and (seg_logits is not None) and (mask is not None):
                pred_mask = torch.sigmoid(seg_logits)
                pred_mask = (pred_mask > 0.5).float()
                pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
                if pred_mask.shape != output.shape:
                    pred_mask = cv.resize(pred_mask.astype(np.uint8), (output.shape[1], output.shape[0]), interpolation=cv.INTER_NEAREST).astype(np.float32)
                
                gt_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
                if gt_mask.shape != output.shape:
                    gt_mask = cv.resize(gt_mask.astype(np.uint8), (output.shape[1], output.shape[0]), interpolation=cv.INTER_NEAREST).astype(np.float32)
                iou, dice = compute_iou_and_dice(pred_mask, gt_mask)
                iou_list.append(iou)
                dice_list.append(dice)

                pd_counts_plots, gt_counts_plots = extract_matched_plot_counts(
                    output, gt_mask, pred_mask, pts,
                )
                all_pd_counts.extend(pd_counts_plots)
                all_gt_counts.extend(gt_counts_plots)
            
            torch.cuda.empty_cache()

    if len(all_pd_counts) > 0:
        mae = compute_mae(all_pd_counts, all_gt_counts)
        rmse = compute_rmse(all_pd_counts, all_gt_counts)
        if len(all_pd_counts) >= 2:
            try:
                r2 = rsquared(all_pd_counts, all_gt_counts)
            except Exception:
                r2 = 0.0
        else:
            r2 = 0.0
    else:
        mae, rmse, r2 = 0.0, 0.0, 0.0
    mean_iou = float(np.mean(iou_list)) if len(iou_list) > 0 else 0.0
    mean_dice = float(np.mean(dice_list)) if len(dice_list) > 0 else 0.0

    msg = 'epoch: {0}, mae: {1:.2f}, rmse: {2:.2f}, r2: {3:.4f}, plots: {4}'.format(
        epoch, mae, rmse, r2, len(all_pd_counts))
    msg += f', IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}'
    print(msg)

    net.val_loss['running_loss'].append(running_loss/(i+1))
    net.val_loss['epoch_loss'].append(mae)
    
    if 'iou' not in net.measure:
        net.measure['iou'] = []
        net.measure['dice'] = []
    net.measure['iou'].append(mean_iou)
    net.measure['dice'].append(mean_dice)
    
    if 'plots_mae' not in net.measure:
        net.measure['plots_mae'] = []
        net.measure['plots_rmse'] = []
        net.measure['plots_r2'] = []
    net.measure['plots_mae'].append(mae)
    net.measure['plots_rmse'].append(rmse)
    net.measure['plots_r2'].append(r2)
    return [], []


def main(lamda, seg_weight=0.0, eval_with_mask=True, resume=True, resume_path=None):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(30)
    torch.manual_seed(30)
    np.random.seed(30)

    train_transforms = transforms.Compose([
        BBRandomCrop(crop_size, crop_num),
        BBRandomFlip(),
        BBNormalize(scale=image_scale, std=image_std, mean=image_mean, train=True),
        BBToTensor(train=True),
        BBZeroPadding(output_stride, train=True)
    ])
    val_transforms = transforms.Compose([
        BBNormalize(scale=image_scale, std=image_std, mean=image_mean, train=False),
        BBToTensor(train=False),
        BBZeroPadding(output_stride, train=False)
    ])
    trainset = BlueberryDataset(
        data_dir="dataset_blueberry_augmented",
        data_list="train.txt",
        train=True,
        transform=train_transforms
    )
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True
    )
    valset = BlueberryDataset(
        data_dir="dataset_blueberry_augmented",
        data_list="test.txt",
        train=False,
        transform=val_transforms
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    net = IntegrateNet(use_deep_supervision=True)
    net = net.cuda()
    criterion = nn.MSELoss().cuda()
    criterion2 = nn.L1Loss().cuda()
    bce_loss = nn.BCEWithLogitsLoss().cuda() if seg_weight > 0 else None

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    trainable_params_m = trainable_params / 1e6
    print("="*60)
    print(f"Model: IntegrateNet - Deep Supervision Only")
    print(f"  - Basic DoubleConv (No residual, No CBAM)")
    print(f"  - Deep Supervision at 1/16 and 1/8 scales")
    print(f"  - Aux weight: 1/16={aux_weight_16}, 1/8={aux_weight_8}")
    print(f"Trainable parameters: {trainable_params_m:.2f} M")
    print("="*60)

    net.train_loss = {'running_loss': [], 'epoch_loss': []}
    net.val_loss = {'running_loss': [], 'epoch_loss': []}
    net.measure = {'plots_mae': [], 'plots_rmse': [], 'plots_r2': []}

    learning_params = [p[1] for p in net.named_parameters()]
    pretrained_params = []
    optimizer = torch.optim.SGD(
        [
            {'params': learning_params},
            {'params': pretrained_params, 'lr': learning_rate / mult},
        ],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    start_epoch = 0
    best_mae = 100000
    best_rmse = 0
    best_r2 = 0
    
    if resume:
        model_dir = "results/models/IntegrateNet_DS"
        if resume_path is None:
            latest_path = os.path.join(model_dir, 'model_latest.pth.tar')
            if os.path.exists(latest_path):
                resume_path = latest_path
            elif os.path.exists(model_dir):
                checkpoints = []
                for f in os.listdir(model_dir):
                    if f.startswith('model_ckpt') and f.endswith('.pth.tar') and f != 'model_ckpt.pth.tar':
                        try:
                            epoch_num = int(f.replace('model_ckpt', '').replace('.pth.tar', ''))
                            checkpoints.append((epoch_num, os.path.join(model_dir, f)))
                        except:
                            pass
                if checkpoints:
                    checkpoints.sort(reverse=True)
                    resume_path = checkpoints[0][1]
        if resume_path and os.path.exists(resume_path):
            start_epoch, history = load_checkpoint(resume_path, net, optimizer)
            if history is not None:
                net.train_loss = history['train_loss']
                net.val_loss = history['val_loss']
                net.measure = history['measure']
                if len(net.measure.get('plots_mae', [])) > 0:
                    best_mae = min(net.measure['plots_mae'])
                    best_idx = net.measure['plots_mae'].index(best_mae)
                    best_rmse = net.measure['plots_rmse'][best_idx]
                    best_r2 = net.measure['plots_r2'][best_idx]

    for epoch in range(start_epoch, num_epoch):
        print("epoch", epoch)
        train(net, train_loader, optimizer, criterion, criterion2, epoch, lamda, seg_weight=seg_weight, bce_loss=bce_loss)
        _, _ = validate(net, val_loader, epoch, criterion, criterion2, lamda, seg_weight=seg_weight, bce_loss=bce_loss, eval_with_mask=eval_with_mask)

        state = {
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss': net.train_loss,
            'val_loss': net.val_loss,
            'measure': net.measure
        }

        if epoch % 100 == 99:
            save_checkpoint(state,"results/models/IntegrateNet_DS", filename='model_ckpt'+str(epoch)+'.pth.tar')

        if net.measure['plots_mae'][-1] <= best_mae:
            save_checkpoint(state, "results/models/IntegrateNet_DS", filename='model_best.pth')
            best_mae = net.measure['plots_mae'][-1]
            best_rmse = net.measure['plots_rmse'][-1]
            best_r2 = net.measure['plots_r2'][-1]
            best_iou = net.measure['iou'][-1]
            best_dice = net.measure['dice'][-1]

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(net.train_loss['epoch_loss'], label='train loss', color='tab:blue')
    ax1.plot(net.val_loss['running_loss'], label='val loss', color='tab:red')
    ax1.legend(loc='upper right')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(net.val_loss['epoch_loss'], label='val mae', color='tab:orange')
    ax2.legend(loc='upper right')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(net.measure['plots_rmse'], label='val rmse', color='tab:red')
    ax3.legend(loc='upper right')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(net.measure['plots_r2'], label='val r2', color='tab:red')
    ax4.legend(loc='upper right')
    plt.savefig("Loss_DS.png", dpi=300)
    plt.close()

    idx = net.measure['plots_mae'].index(min(net.measure['plots_mae']))
    print('lambda', lamda)
    print("the best result is: mae:", best_mae, 'rmse', best_rmse, 'r2', best_r2, 'IoU', best_iou, 'Dice', best_dice, "epoch", idx)


if __name__ == "__main__":
    main(0.5, seg_weight=1.0, eval_with_mask=True, resume=False)
