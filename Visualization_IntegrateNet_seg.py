import argparse
import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
P2P_DIR = os.path.join(ROOT_DIR, 'CrowdCounting-P2PNet-main')
DM_DIR = os.path.join(ROOT_DIR, 'DM-Count-master')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from error import compute_mae, compute_rmse, rsquared


def _clear_models_cache():
    """清除已缓存的 models 模块，避免 DM-Count models.py 与 P2PNet models/ 包冲突。"""
    for key in list(sys.modules):
        if key == 'models' or key.startswith('models.'):
            del sys.modules[key]


def _prep_sys_path(preferred_dir):
    """将 preferred_dir 置于 sys.path 首位，避免 models 包名冲突。"""
    if preferred_dir in sys.path:
        sys.path.remove(preferred_dir)
    sys.path.insert(0, preferred_dir)


def _import_dm_modules():
    """DM-Count 依赖（models.py 与 P2PNet 的 models/ 包同名，需单独路径）。"""
    _prep_sys_path(DM_DIR)
    _clear_models_cache()
    from datasets.blueberry import BlueberryDMCount
    from models import vgg19
    from traval_dmcount import SegHead, VGGWithSeg
    return BlueberryDMCount, vgg19, SegHead, VGGWithSeg


def _import_p2pnet_modules():
    """P2PNet 依赖（仅在 P2PNet 可视化时使用）。"""
    _prep_sys_path(P2P_DIR)
    _clear_models_cache()
    from crowd_datasets.blueberry.blueberry import BlueberryP2PNet
    from models.backbone import build_backbone
    from traval_p2pnet import P2PNetWithSeg, _merge_points_nms
    return BlueberryP2PNet, build_backbone, P2PNetWithSeg, _merge_points_nms

# P2PNet 归一化（BlueberryP2PNet 使用 ImageNet）
P2P_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
P2P_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EVAL_THRESHOLD = 0.5
NMS_DIST = 8.0
SW_MAX_SIZE = 3200
SW_STRIDE = 512
SW_CROP = 768

# 散点图样式（与 IntegrateNet-MTSC 参考图一致）
SCATTER_AXIS_MAX = 45
SCATTER_MIN_GT = 0
SCATTER_MARKER_SIZE = 135
SCATTER_MARKER_COLOR = '#3232d0'   # 略深于 RGB(70, 70, 240)
SCATTER_MARKER_EDGE_COLOR = '#1a1a1a'
SCATTER_MARKER_EDGE_WIDTH = 1.0
SCATTER_MARKER_ALPHA = 0.82
SCATTER_LABEL_FONTSIZE = 18
SCATTER_TICK_FONTSIZE = 16
SCATTER_TITLE_FONTSIZE = 20
SCATTER_R2_FONTSIZE = 16
SCATTER_LEGEND_FONTSIZE = 14
SCATTER_FONT = 'Times New Roman'


def denorm_p2p_image(input_tensor):
    """反归一化为 BGR uint8。"""
    img_np = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * P2P_STD + P2P_MEAN
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return cv.cvtColor(img_np, cv.COLOR_RGB2BGR)


def _align_mask_pair(pred_mask, gt_mask, orig_h, orig_w):
    orig_h, orig_w = int(orig_h), int(orig_w)
    pred_mask = pred_mask[:orig_h, :orig_w]
    gt_mask = gt_mask[:orig_h, :orig_w]
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv.resize(
            gt_mask.astype(np.uint8), (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv.INTER_NEAREST,
        ).astype(np.float32)
    return pred_mask, gt_mask


def _filter_points_to_orig(pred_pts, orig_h, orig_w):
    if len(pred_pts) == 0:
        return pred_pts
    keep = ((pred_pts[:, 0] >= 0) & (pred_pts[:, 0] < orig_w) &
            (pred_pts[:, 1] >= 0) & (pred_pts[:, 1] < orig_h))
    return pred_pts[keep]


def infer_p2pnet_full(
    net,
    input_full,
    eval_threshold=EVAL_THRESHOLD,
    nms_dist=NMS_DIST,
    max_size=SW_MAX_SIZE,
    stride=SW_STRIDE,
    crop_size=SW_CROP,
):
    """全图 P2PNet 推理，返回 seg_logits、pred_pts_np（原图坐标）。"""
    _, _, _, _merge_points_nms = _import_p2pnet_modules()
    _, _, h, w = input_full.shape
    device = input_full.device

    if h > max_size or w > max_size:
        seg_logits_full = torch.zeros((1, 1, h, w), device=device)
        seg_count_map = torch.zeros((1, 1, h, w), device=device)
        all_points, all_scores = [], []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + crop_size, h)
                x_end = min(x + crop_size, w)
                y_start = max(0, y_end - crop_size)
                x_start = max(0, x_end - crop_size)

                patch = input_full[:, :, y_start:y_end, x_start:x_end]
                outputs = net(patch)

                seg_p = outputs['seg_logits']
                seg_logits_full[:, :, y_start:y_end, x_start:x_end] += seg_p
                seg_count_map[:, :, y_start:y_end, x_start:x_end] += 1.0

                scores = F.softmax(outputs['pred_logits'], dim=-1)[0, :, 1]
                pred_pts = outputs['pred_points'][0]
                keep = scores > eval_threshold
                if keep.any():
                    pts_np = pred_pts[keep].detach().cpu().numpy()
                    pts_np[:, 0] += x_start
                    pts_np[:, 1] += y_start
                    all_points.append(pts_np)
                    all_scores.append(scores[keep].detach().cpu().numpy())

        seg_logits = seg_logits_full / seg_count_map.clamp(min=1.0)
        if all_points:
            pts_cat = np.concatenate(all_points, axis=0)
            scr_cat = np.concatenate(all_scores, axis=0)
            pred_pts_np = _merge_points_nms(pts_cat, scr_cat, dist_thresh=nms_dist)
        else:
            pred_pts_np = np.zeros((0, 2), dtype=np.float32)
    else:
        outputs = net(input_full)
        seg_logits = outputs['seg_logits']
        scores = F.softmax(outputs['pred_logits'], dim=-1)[0, :, 1]
        pred_pts = outputs['pred_points'][0]
        keep = scores > eval_threshold
        if keep.any():
            pts_np = pred_pts[keep].detach().cpu().numpy()
            scr_np = scores[keep].detach().cpu().numpy()
            pred_pts_np = _merge_points_nms(pts_np, scr_np, dist_thresh=nms_dist)
        else:
            pred_pts_np = np.zeros((0, 2), dtype=np.float32)

    return seg_logits, pred_pts_np


class P2PNetValDataset(Dataset):
    def __init__(self, data_dir, data_list, crop_size=256):
        BlueberryP2PNet, _, _, _ = _import_p2pnet_modules()
        self.ds = BlueberryP2PNet(
            data_dir=data_dir, data_list=data_list,
            crop_size=crop_size, num_patch=4, train=False, flip=False,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_t, targets = self.ds[idx]
        return img_t, targets[0]


def p2p_collate(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    return imgs, [b[1] for b in batch]


def _image_stem(name):
    if isinstance(name, torch.Tensor):
        name = str(name.item())
    return os.path.splitext(str(name))[0]


def infer_dmcount_density_map(net, input_full, max_size=SW_MAX_SIZE, stride=SW_STRIDE, crop_size=SW_CROP):
    """DM-Count 全图密度推理（与 traval_dmcount.validate Part 2 一致，含 sum 保持）。"""
    _, _, h, w = input_full.shape

    if h > max_size or w > max_size:
        h_low, w_low = h // 8, w // 8
        output_full_low = torch.zeros((1, 1, h_low, w_low), device=input_full.device)
        seg_logits_full = torch.zeros((1, 1, h, w), device=input_full.device)
        count_map_low = torch.zeros((1, 1, h_low, w_low), device=input_full.device)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + crop_size, h)
                x_end = min(x + crop_size, w)
                y_start = max(0, y_end - crop_size)
                x_start = max(0, x_end - crop_size)
                y_low_s, y_low_e = y_start // 8, y_end // 8
                x_low_s, x_low_e = x_start // 8, x_end // 8

                patch = input_full[:, :, y_start:y_end, x_start:x_end]
                out_p, _, seg_p = net(patch)
                out_p_low = F.interpolate(
                    out_p, size=(y_low_e - y_low_s, x_low_e - x_low_s),
                    mode='bilinear', align_corners=False,
                )
                output_full_low[:, :, y_low_s:y_low_e, x_low_s:x_low_e] += out_p_low
                count_map_low[:, :, y_low_s:y_low_e, x_low_s:x_low_e] += 1.0

                seg_p_up = F.interpolate(
                    seg_p, size=(y_end - y_start, x_end - x_start),
                    mode='bilinear', align_corners=False,
                )
                seg_logits_full[:, :, y_start:y_end, x_start:x_end] += seg_p_up

        outputs = output_full_low / count_map_low.clamp(min=1.0)
        output_for_metrics = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=False)

        seg_logits_count = torch.zeros((1, 1, h, w), device=input_full.device)
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end_s = min(y + crop_size, h)
                x_end_s = min(x + crop_size, w)
                y_start_s = max(0, y_end_s - crop_size)
                x_start_s = max(0, x_end_s - crop_size)
                seg_logits_count[:, :, y_start_s:y_end_s, x_start_s:x_end_s] += 1.0
        seg_logits = seg_logits_full / seg_logits_count.clamp(min=1.0)
    else:
        outputs, _, seg_logits = net(input_full)
        output_for_metrics = F.interpolate(
            outputs, size=(h, w), mode='bilinear', align_corners=False,
        )
        seg_logits = F.interpolate(
            seg_logits, size=(h, w), mode='bilinear', align_corners=False,
        )

    output_np_lowres = outputs.squeeze().detach().cpu().numpy()
    output_np_lowres = np.clip(output_np_lowres, 0, None)
    sum_lowres = float(output_np_lowres.sum())

    output_np = output_for_metrics.squeeze().detach().cpu().numpy()
    output_np = np.nan_to_num(output_np, nan=0.0, posinf=0.0, neginf=0.0)
    output_np = np.clip(output_np, 0, None)
    sum_highres = float(output_np.sum())
    if sum_highres > 0:
        output_np = output_np * (sum_lowres / sum_highres)

    return output_np, seg_logits


def extract_plot_counts(output_np, gt_mask, points):
    """从密度图 + GT mask 提取各地块 pred/gt 计数。"""
    gt_mask_np = gt_mask.squeeze(0).squeeze(0).cpu().numpy() if isinstance(gt_mask, torch.Tensor) else np.array(gt_mask)
    if gt_mask_np.ndim == 3:
        gt_mask_np = gt_mask_np[0]
    if gt_mask_np.shape != output_np.shape:
        gt_mask_np = cv.resize(
            gt_mask_np.astype(np.uint8),
            (output_np.shape[1], output_np.shape[0]),
            interpolation=cv.INTER_NEAREST,
        ).astype(np.float32)

    gt_bin = (gt_mask_np > 0.5).astype(np.uint8)
    num_labels, labels = cv.connectedComponents(gt_bin, connectivity=8)

    if points is not None:
        if isinstance(points, torch.Tensor):
            pts_arr = points.cpu().numpy()
        else:
            pts_arr = np.array(points[0] if isinstance(points, (list, tuple)) else points)
        if pts_arr.ndim == 3 and pts_arr.shape[0] == 1:
            pts_arr = pts_arr.squeeze(0)
        if pts_arr.size > 0 and pts_arr.ndim == 2 and pts_arr.shape[1] == 2:
            xs = np.clip(np.rint(pts_arr[:, 0]).astype(np.int32), 0, gt_bin.shape[1] - 1)
            ys = np.clip(np.rint(pts_arr[:, 1]).astype(np.int32), 0, gt_bin.shape[0] - 1)
        else:
            xs = np.empty((0,), dtype=np.int32)
            ys = np.empty((0,), dtype=np.int32)
    else:
        xs = np.empty((0,), dtype=np.int32)
        ys = np.empty((0,), dtype=np.int32)

    pd_counts, gt_counts = [], []
    for lab in range(1, num_labels):
        region = (labels == lab).astype(np.float32)
        pd_counts.append(float((output_np * region).sum()))
        if xs.size > 0:
            gt_counts.append(int((region[ys, xs] > 0.5).sum()))
        else:
            gt_counts.append(0)
    return pd_counts, gt_counts


def extract_p2p_plot_counts(pred_pts_np, gt_mask, gt_points, orig_h, orig_w):
    """P2PNet：按 GT mask 连通域统计 pred/gt 点数量（与 traval_p2pnet.validate 一致）。"""
    gt_mask_t = gt_mask
    gt_mask_np = gt_mask_t[0].cpu().numpy() if isinstance(gt_mask_t, torch.Tensor) \
        else np.array(gt_mask_t[0])
    if gt_mask_np.ndim == 3:
        gt_mask_np = gt_mask_np[0]

    orig_h, orig_w = int(orig_h), int(orig_w)
    gt_mask_np = gt_mask_np[:orig_h, :orig_w]

    if isinstance(gt_points, torch.Tensor):
        gt_pts_np = gt_points.cpu().numpy()
    elif gt_points is not None:
        gt_pts_np = np.array(gt_points)
    else:
        gt_pts_np = np.zeros((0, 2), dtype=np.float32)
    if gt_pts_np.ndim == 3 and gt_pts_np.shape[0] == 1:
        gt_pts_np = gt_pts_np.squeeze(0)

    pred_pts_np = np.asarray(pred_pts_np, dtype=np.float32)
    if len(pred_pts_np) > 0:
        keep = ((pred_pts_np[:, 0] >= 0) & (pred_pts_np[:, 0] < orig_w) &
                (pred_pts_np[:, 1] >= 0) & (pred_pts_np[:, 1] < orig_h))
        pred_pts_np = pred_pts_np[keep]

    gt_bin = (gt_mask_np > 0.5).astype(np.uint8)
    num_labels, labels = cv.connectedComponents(gt_bin, connectivity=8)

    pd_counts, gt_counts = [], []
    for lab in range(1, num_labels):
        region = (labels == lab)

        pd_cnt = 0
        if len(pred_pts_np) > 0:
            px = np.clip(np.rint(pred_pts_np[:, 0]).astype(int), 0, region.shape[1] - 1)
            py = np.clip(np.rint(pred_pts_np[:, 1]).astype(int), 0, region.shape[0] - 1)
            pd_cnt = int(region[py, px].sum())

        gt_cnt = 0
        if len(gt_pts_np) > 0 and gt_pts_np.ndim == 2 and gt_pts_np.shape[1] == 2:
            gx = np.clip(np.rint(gt_pts_np[:, 0]).astype(int), 0, region.shape[1] - 1)
            gy = np.clip(np.rint(gt_pts_np[:, 1]).astype(int), 0, region.shape[0] - 1)
            gt_cnt = int(region[gy, gx].sum())

        pd_counts.append(pd_cnt)
        gt_counts.append(gt_cnt)
    return pd_counts, gt_counts


def collect_dmcount_plot_counts(net, val_loader):
    """在测试集上收集所有地块级 pred/gt 计数。"""
    all_pd, all_gt = [], []
    with torch.no_grad():
        for sample in val_loader:
            image = sample['image'].cuda()
            mask = sample.get('mask', None)
            pts = sample.get('points', None)
            if isinstance(pts, (list, tuple)) and len(pts) == 1:
                pts = pts[0]

            output_np, _ = infer_dmcount_density_map(net, image)
            if mask is None:
                continue
            pd_plots, gt_plots = extract_plot_counts(output_np, mask, pts)
            all_pd.extend(pd_plots)
            all_gt.extend(gt_plots)
            torch.cuda.empty_cache()
    return all_gt, all_pd


def collect_p2pnet_plot_counts(net, val_loader, eval_threshold=EVAL_THRESHOLD, nms_dist=NMS_DIST):
    """P2PNet 测试集地块级 pred/gt 计数。"""
    all_pd, all_gt = [], []
    with torch.no_grad():
        for input_image, targets in val_loader:
            target = targets[0]
            input_cuda = input_image.cuda()
            orig_h = int(target['orig_h']) if not isinstance(target['orig_h'], torch.Tensor) \
                else int(target['orig_h'].item())
            orig_w = int(target['orig_w']) if not isinstance(target['orig_w'], torch.Tensor) \
                else int(target['orig_w'].item())

            _, pred_pts_np = infer_p2pnet_full(
                net, input_cuda, eval_threshold=eval_threshold, nms_dist=nms_dist,
            )
            pred_pts_np = _filter_points_to_orig(pred_pts_np, orig_h, orig_w)

            gt_mask = target.get('mask')
            if gt_mask is None:
                continue
            points = target.get('point')
            pd_plots, gt_plots = extract_p2p_plot_counts(
                pred_pts_np, gt_mask, points, orig_h, orig_w,
            )
            all_pd.extend(pd_plots)
            all_gt.extend(gt_plots)
            torch.cuda.empty_cache()
    return all_gt, all_pd


def plot_count_scatter(gt_counts, pd_counts, model_name, save_path,
                       axis_max=SCATTER_AXIS_MAX, min_gt=SCATTER_MIN_GT):
    """绘制地块级 GT vs Pred 散点图（样式对齐 IntegrateNet-MTSC 参考图）。"""
    gt = np.asarray(gt_counts, dtype=np.float64)
    pd = np.asarray(pd_counts, dtype=np.float64)

    valid = np.isfinite(gt) & np.isfinite(pd)
    if min_gt is not None:
        valid &= (gt >= min_gt)
    gt, pd = gt[valid], pd[valid]

    r2 = rsquared(pd, gt) if len(gt) >= 2 else 0.0
    mae = compute_mae(pd, gt) if len(gt) > 0 else 0.0
    rmse = compute_rmse(pd, gt) if len(gt) > 0 else 0.0

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [SCATTER_FONT, 'Times', 'DejaVu Serif'],
        'mathtext.default': 'regular',
    })

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(
        gt, pd,
        s=SCATTER_MARKER_SIZE,
        c=SCATTER_MARKER_COLOR,
        alpha=SCATTER_MARKER_ALPHA,
        edgecolors=SCATTER_MARKER_EDGE_COLOR,
        linewidths=SCATTER_MARKER_EDGE_WIDTH,
        zorder=3,
    )

    ax.plot(
        [0, axis_max], [0, axis_max],
        color='red', linestyle='--', linewidth=2,
        label='y = x', zorder=2,
    )

    tick_vals = list(range(0, int(axis_max) + 1, 10))
    if tick_vals[-1] > axis_max:
        tick_vals = tick_vals[:-1]
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(
        'Ground Truth Count', fontsize=SCATTER_LABEL_FONTSIZE,
        fontweight='bold', fontname=SCATTER_FONT,
    )
    ax.set_ylabel(
        'Predicted Count', fontsize=SCATTER_LABEL_FONTSIZE,
        fontweight='bold', fontname=SCATTER_FONT,
    )
    ax.set_title(
        model_name, fontsize=SCATTER_TITLE_FONTSIZE,
        fontweight='bold', pad=14, fontname=SCATTER_FONT,
    )

    ax.text(
        0.05, 0.95, f'R\u00b2 = {r2}',
        transform=ax.transAxes, fontsize=SCATTER_R2_FONTSIZE, va='top',
        fontname=SCATTER_FONT, fontstyle='normal',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1),
    )

    leg = ax.legend(
        loc='lower right', frameon=True, fancybox=False, framealpha=1.0,
        facecolor='none', edgecolor='#cccccc',
        prop={'family': SCATTER_FONT, 'size': SCATTER_LEGEND_FONTSIZE, 'style': 'normal'},
    )
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor('#cccccc')

    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.55, color='0.75')
    ax.tick_params(axis='both', which='major', labelsize=SCATTER_TICK_FONTSIZE, width=0.6, length=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(SCATTER_FONT)
        label.set_fontstyle('normal')
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.savefig(
        save_path, dpi=300,
        bbox_inches='tight', pad_inches=0.08,
        facecolor='white', edgecolor='none',
    )
    plt.close(fig)

    print(f"Saved scatter plot: {save_path}")
    print(f"  plots={len(gt)} (min_gt>={min_gt})  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return r2, mae, rmse


def visualize_prediction(
    image, pred_mask, gt_mask, points,
    save_path_gt, save_path_pred, pred_points=None, overlay_map=None,
):
    """GT/Pred 可视化。overlay_map 为密度图时 Pred 叠加热力图；P2PNet 无密度图则不叠加。"""
    pts_arr = None
    if points is not None and len(points) > 0:
        if isinstance(points, torch.Tensor):
            pts_arr = points.cpu().numpy()
        elif isinstance(points, np.ndarray):
            pts_arr = points
        else:
            pts_arr = np.array(points)
        if pts_arr.ndim == 3 and pts_arr.shape[0] == 1:
            pts_arr = pts_arr.squeeze(0)
        if not (pts_arr.size > 0 and pts_arr.ndim == 2 and pts_arr.shape[1] == 2):
            pts_arr = None

    gt_mask_bin = (gt_mask > 0.5).astype(np.uint8)
    num_gt_labels, gt_labels = cv.connectedComponents(gt_mask_bin, connectivity=8)

    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
    num_pred_labels, pred_labels = cv.connectedComponents(pred_mask_bin, connectivity=8)

    image_with_gt = image.copy()
    border_color_red = (0, 0, 255)

    for lab in range(1, num_gt_labels):
        region = (gt_labels == lab).astype(np.uint8)
        contours, _ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(image_with_gt, contours, -1, border_color_red, 5)

        gt_count = 0
        if pts_arr is not None:
            xs = np.clip(np.rint(pts_arr[:, 0]).astype(np.int32), 0, region.shape[1] - 1)
            ys = np.clip(np.rint(pts_arr[:, 1]).astype(np.int32), 0, region.shape[0] - 1)
            gt_count = int(region[ys, xs].sum())

        M = cv.moments(region)
        if M["m00"] != 0 and gt_count > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            text = f"GT:{gt_count}"
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale, thickness, padding = 4, 9, 15
            (tw, th), _ = cv.getTextSize(text, font, font_scale, thickness)
            half_w, half_h = tw // 2 + padding, th // 2 + padding
            H, W = image_with_gt.shape[:2]
            cx = int(np.clip(cx, half_w, W - half_w - 1))
            cy = int(np.clip(cy, half_h, H - half_h - 1))
            cv.rectangle(image_with_gt,
                         (cx - half_w, cy - half_h), (cx + half_w, cy + half_h),
                         (255, 255, 255), -1)
            cv.putText(image_with_gt, text, (cx - tw // 2, cy + th // 2),
                       font, font_scale, (255, 0, 0), thickness)

    if pts_arr is not None:
        for (x, y) in pts_arr:
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]:
                cv.circle(image_with_gt, (cx, cy), radius=16, color=(0, 0, 255), thickness=2)

    fig_gt = plt.figure(figsize=(12, 8))
    plt.imshow(cv.cvtColor(image_with_gt, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path_gt, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Pred：有密度图时叠加热力图，否则仅原图 + 轮廓/点
    if overlay_map is not None:
        overlay_norm = (overlay_map - overlay_map.min()) / (overlay_map.max() - overlay_map.min() + 1e-8)
        overlay_cmap = (plt.cm.jet(overlay_norm)[:, :, :3] * 255).astype(np.uint8)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        alpha = 0.3
        image_with_overlay = (
            image_rgb.astype(np.float32) * (1 - alpha) +
            overlay_cmap.astype(np.float32) * alpha
        ).astype(np.uint8)
        image_with_pred = cv.cvtColor(image_with_overlay, cv.COLOR_RGB2BGR)
    else:
        image_with_pred = image.copy()

    for lab in range(1, num_pred_labels):
        region = (pred_labels == lab).astype(np.uint8)

        if pred_points is not None and len(pred_points) > 0:
            px = np.clip(np.rint(pred_points[:, 0]).astype(int), 0, region.shape[1] - 1)
            py = np.clip(np.rint(pred_points[:, 1]).astype(int), 0, region.shape[0] - 1)
            pred_count = int(region[py, px].sum())
        elif overlay_map is not None:
            pred_count = int((overlay_map * region).sum())
        else:
            pred_count = 0

        if pred_count >= 1:
            contours, _ = cv.findContours(region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(image_with_pred, contours, -1, border_color_red, 5)

            M = cv.moments(region)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                text = f"Pred:{pred_count}"
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale, thickness, padding = 4, 9, 15
                (tw, th), _ = cv.getTextSize(text, font, font_scale, thickness)
                half_w, half_h = tw // 2 + padding, th // 2 + padding
                H, W = image_with_pred.shape[:2]
                cx = int(np.clip(cx, half_w, W - half_w - 1))
                cy = int(np.clip(cy, half_h, H - half_h - 1))
                cv.rectangle(image_with_pred,
                             (cx - half_w, cy - half_h), (cx + half_w, cy + half_h),
                             (255, 255, 255), -1)
                cv.putText(image_with_pred, text, (cx - tw // 2, cy + th // 2),
                           font, font_scale, (255, 0, 0), thickness)

    if pred_points is not None and len(pred_points) > 0:
        for (x, y) in pred_points:
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]:
                cv.circle(image_with_pred, (cx, cy), radius=18, color=(255, 0, 0), thickness=-1)

    fig_pred = plt.figure(figsize=(12, 8))
    plt.imshow(cv.cvtColor(image_with_pred, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path_pred, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    return num_pred_labels - 1


if __name__ == "__main__":
    # ── P2PNet 地块级散点图 ──────────────────────────────────────────────────
    data_dir = "dataset_blueberry_augmented"
    test_list = "final_test.txt"
    model_path = "results/models/P2PNet/model_best.pth"
    output_dir = "results/visualization_P2PNet"
    model_title = "P2PNet"

    os.makedirs(output_dir, exist_ok=True)

    valset = P2PNetValDataset(data_dir=data_dir, data_list=test_list, crop_size=256)
    val_loader = DataLoader(
        valset, batch_size=1, shuffle=False,
        collate_fn=p2p_collate, num_workers=0,
    )

    _, build_backbone, P2PNetWithSeg, _ = _import_p2pnet_modules()
    bb_args = argparse.Namespace(backbone='vgg16_bn')
    net = P2PNetWithSeg(build_backbone(bb_args), row=2, line=2)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[7:] if k.startswith('module.') else k: v
                      for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=False)
    net = net.cuda().eval()

    print("=" * 60)
    print("P2PNet Scatter Plot (per-plot GT vs Pred)")
    print(f"Model : {model_path}")
    print(f"Data  : {data_dir}/{test_list}  ({len(valset)} images)")
    print(f"Save  : {output_dir}/scatter_plot.png")
    print(f"Thr   : eval_threshold=0.35, nms_dist={NMS_DIST}")
    print("=" * 60)

    gt_counts, pd_counts = collect_p2pnet_plot_counts(
        net, val_loader, eval_threshold=0.35, nms_dist=NMS_DIST,
    )
    save_path = f"{output_dir}/scatter_plot.png"
    plot_count_scatter(gt_counts, pd_counts, model_title, save_path)

    print("=" * 60)
    print(f"Done. Scatter plot saved to {save_path}")
    print("=" * 60)

    # ── DM-Count 地块级散点图（已注释，需要时取消注释） ───────────────────────
    # data_dir = "dataset_blueberry_augmented"
    # test_list = "final_test.txt"
    # model_path = "results/models/DMCount/model_best.pth"
    # output_dir = "results/visualization_DMCount"
    # model_title = "DM-Count"
    #
    # os.makedirs(output_dir, exist_ok=True)
    #
    # BlueberryDMCount, vgg19, SegHead, VGGWithSeg = _import_dm_modules()
    # valset = BlueberryDMCount(
    #     data_dir=data_dir, data_list=test_list,
    #     crop_size=256, downsample_ratio=8, method='val',
    # )
    # val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    #
    # net = VGGWithSeg(vgg19(), SegHead(in_channels=128)).cuda()
    # checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    # state_dict = checkpoint.get('state_dict') or checkpoint.get('model') or checkpoint
    # if any(k.startswith('module.') for k in state_dict):
    #     state_dict = {k[7:] if k.startswith('module.') else k: v
    #                   for k, v in state_dict.items()}
    # net.load_state_dict(state_dict, strict=False)
    # net.eval()
    #
    # gt_counts, pd_counts = collect_dmcount_plot_counts(net, val_loader)
    # plot_count_scatter(gt_counts, pd_counts, model_title, f"{output_dir}/scatter_plot.png")

    # ── 原 P2PNet GT/Pred 可视化（已注释，需要时取消注释） ─────────────────────
    # data_dir = "dataset_blueberry_augmented"
    # test_list = "final_test.txt"
    # model_path = "results/models/P2PNet/model_best.pth"
    # output_dir = "results/visualization_P2PNet"
    # ONLY_IMAGE = "DJI_0249_crop_9_orig"   # 例如 "DJI_0249_crop_9_orig"
    #
    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(f"{output_dir}/prediction", exist_ok=True)
    #
    # valset = P2PNetValDataset(data_dir=data_dir, data_list=test_list, crop_size=256)
    # val_loader = DataLoader(
    #     valset, batch_size=1, shuffle=False,
    #     collate_fn=p2p_collate, num_workers=0,
    # )
    #
    # bb_args = argparse.Namespace(backbone='vgg16_bn')
    # net = P2PNetWithSeg(build_backbone(bb_args), row=2, line=2)
    # checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    # state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    # if any(k.startswith('module.') for k in state_dict):
    #     state_dict = {k[7:] if k.startswith('module.') else k: v
    #                   for k, v in state_dict.items()}
    # net.load_state_dict(state_dict, strict=False)
    # net = net.cuda().eval()
    #
    # print("=" * 60)
    # print("Visualizing P2PNet Results")
    # print(f"Model : {model_path}")
    # print(f"Data  : {data_dir}/{test_list}  ({len(valset)} images)")
    # print(f"Save  : {output_dir}/prediction/")
    # print(f"Thr   : eval_threshold={EVAL_THRESHOLD}, nms_dist={NMS_DIST}")
    # print("=" * 60)
    #
    # total_samples = len(valset)
    # processed = 0
    # with torch.no_grad():
    #     for i, (input_image, targets) in enumerate(val_loader):
    #         target = targets[0]
    #         image_name = target['filename']
    #         if isinstance(image_name, torch.Tensor):
    #             image_name = str(image_name.item())
    #         if ONLY_IMAGE is not None and _image_stem(image_name) != _image_stem(ONLY_IMAGE):
    #             continue
    #         processed += 1
    #         print(f"Processing {i + 1}/{total_samples}: {image_name}")
    #
    #         input_cuda = input_image.cuda()
    #         orig_h = int(target['orig_h']) if not isinstance(target['orig_h'], torch.Tensor) \
    #             else int(target['orig_h'].item())
    #         orig_w = int(target['orig_w']) if not isinstance(target['orig_w'], torch.Tensor) \
    #             else int(target['orig_w'].item())
    #
    #         seg_logits, pred_pts_np = infer_p2pnet_full(net, input_cuda)
    #         pred_pts_np = _filter_points_to_orig(pred_pts_np, orig_h, orig_w)
    #
    #         img_np = denorm_p2p_image(input_cuda)[:orig_h, :orig_w]
    #
    #         seg_prob = torch.sigmoid(seg_logits[0, 0]).float().cpu().numpy()
    #         pred_mask_np = (seg_prob > 0.5).astype(np.float32)
    #
    #         gt_mask_t = target['mask']
    #         gt_mask_np = gt_mask_t[0].cpu().numpy() if isinstance(gt_mask_t, torch.Tensor) \
    #             else np.array(gt_mask_t[0])
    #
    #         pred_mask_np, gt_mask_np = _align_mask_pair(
    #             pred_mask_np, gt_mask_np, orig_h, orig_w)
    #
    #         points = target.get('point')
    #         gt_cnt = len(points) if isinstance(points, torch.Tensor) else len(points or [])
    #         print(f"  pred_pts={len(pred_pts_np)}  gt_pts={gt_cnt}")
    #
    #         gt_save_path = f"{output_dir}/prediction/{image_name}_gt.png"
    #         pred_save_path = f"{output_dir}/prediction/{image_name}_pred.png"
    #         visualize_prediction(
    #             img_np, pred_mask_np, gt_mask_np, points,
    #             gt_save_path, pred_save_path, pred_points=pred_pts_np,
    #         )
    #
    # print("=" * 60)
    # if ONLY_IMAGE is not None and processed == 0:
    #     print(f"Warning: ONLY_IMAGE='{ONLY_IMAGE}' matched no samples in {test_list}.")
    # print(f"Done. {processed} image(s) saved to {output_dir}/prediction/")
    # print("=" * 60)
