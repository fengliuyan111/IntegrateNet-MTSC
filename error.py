import scipy
import numpy as np
import math
import cv2 as cv
from scipy.ndimage import gaussian_filter, morphology
from skimage.measure import label, regionprops
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np


def compute_mae(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mae = np.mean(np.abs(diff))
    return mae


def compute_rmse(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mse = np.sqrt(np.mean((diff ** 2)))
    return mse


def rsquared(pd, gt):
    """ Return R^2 where x and y are array-like."""
    return r2_score(gt,pd)


def compute_iou_and_dice(pred_mask, gt_mask, eps=1e-6):
    """
    pred_mask, gt_mask: numpy arrays of shape (H, W), values in {0,1} or [0,1].
    Returns (iou, dice).
    """
    pred = (pred_mask > 0.5).astype(np.float32)
    gt = (gt_mask > 0.5).astype(np.float32)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    iou = inter / (union + eps)
    dice = (2.0 * inter) / (pred.sum() + gt.sum() + eps)
    return float(iou), float(dice)


def _prepare_point_coords(pts, h, w):
    """Extract clipped (xs, ys) integer coordinates from point annotations."""
    if pts is None:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    if hasattr(pts, 'cpu'):
        pts_arr = pts.cpu().numpy()
    else:
        pts_arr = np.array(pts[0] if isinstance(pts, (list, tuple)) else pts)
    if pts_arr.ndim == 3 and pts_arr.shape[0] == 1:
        pts_arr = pts_arr.squeeze(0)
    if pts_arr.size > 0 and pts_arr.ndim == 2 and pts_arr.shape[1] == 2:
        xs = np.clip(np.rint(pts_arr[:, 0]).astype(np.int32), 0, w - 1)
        ys = np.clip(np.rint(pts_arr[:, 1]).astype(np.int32), 0, h - 1)
        return xs, ys
    return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)


def _align_mask_to_shape(mask, target_shape):
    """Resize mask to (H, W) if needed."""
    if hasattr(mask, 'cpu'):
        mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()
    else:
        mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    if mask_np.shape != target_shape:
        mask_np = cv.resize(
            mask_np.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv.INTER_NEAREST,
        ).astype(np.float32)
    return mask_np


def extract_matched_plot_counts(output_np, gt_mask, pred_mask, points=None):
    """
    按地块计数（评估用，不参与训练反传）：
    1. 对 GT/预测分割掩码分别做连通域分析；
    2. 每个预测地块按最大重叠原则归属到一个 GT 地块（多对一，允许 Pred>GT）；
    3. 对每个 GT 地块：预测数 = 所有归属该 GT 的预测地块区域内密度积分之和；
       无归属预测地块的 GT 预测数为 0；与任何 GT 无重叠的预测地块不计入。

    说明：训练损失仍为全图像素级密度/分割监督，与此地块匹配计数无关；
    本函数仅用于验证集 MAE/RMSE/R² 与最优模型选择。
    """
    target_shape = output_np.shape[:2]
    gt_mask_np = _align_mask_to_shape(gt_mask, target_shape)
    pred_mask_np = _align_mask_to_shape(pred_mask, target_shape)

    gt_bin = (gt_mask_np > 0.5).astype(np.uint8)
    pred_bin = (pred_mask_np > 0.5).astype(np.uint8)
    num_gt, gt_labels = cv.connectedComponents(gt_bin, connectivity=8)
    num_pred, pred_labels = cv.connectedComponents(pred_bin, connectivity=8)

    h, w = gt_bin.shape
    xs, ys = _prepare_point_coords(points, h, w)

    # 每个预测地块 → 重叠最大的 GT 地块（多预测地块可归属同一 GT）
    gt_to_pred_labs = {gt_lab: [] for gt_lab in range(1, num_gt)}
    for pred_lab in range(1, num_pred):
        pred_region = (pred_labels == pred_lab)
        best_overlap = 0
        best_gt_lab = -1
        for gt_lab in range(1, num_gt):
            gt_region = (gt_labels == gt_lab)
            overlap = int(np.logical_and(pred_region, gt_region).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_gt_lab = gt_lab
        if best_overlap > 0:
            gt_to_pred_labs[best_gt_lab].append(pred_lab)

    pd_counts, gt_counts = [], []
    for gt_lab in range(1, num_gt):
        gt_region = (gt_labels == gt_lab)
        gt_region_f = gt_region.astype(np.float32)

        pred_cnt = 0.0
        for pred_lab in gt_to_pred_labs[gt_lab]:
            pred_region = (pred_labels == pred_lab).astype(np.float32)
            pred_cnt += float((output_np * pred_region).sum())
        pd_counts.append(pred_cnt)

        if xs.size > 0:
            gt_counts.append(int((gt_region_f[ys, xs] > 0.5).sum()))
        else:
            gt_counts.append(0)
    return pd_counts, gt_counts


def dense_sample2d(x, sx, stride):
    (h, w) = x.shape[:2]
    # idx_img = np.array([i for i in range(h*w)]).reshape(h,w)
    idx_img = np.zeros((h, w), dtype=float)

    th = [i for i in range(0, h - sx + 1, stride)]
    tw = [j for j in range(0, w - sx + 1, stride)]
    norm_vec = np.zeros(len(th) * len(tw))

    for i in th:
        for j in tw:
            idx_img[i:i + sx, j:j + sx] = idx_img[i:i + sx, j:j + sx] + 1

    # # plot redundancy map
    # import os
    # import matplotlib.pyplot as plt
    # cmap = plt.cm.get_cmap('hot')
    # idx_img = idx_img / (idx_img.max())
    # idx_img = cmap(idx_img) * 255.
    # plt.figure()
    # plt.imshow(idx_img.astype(np.uint8))
    # plt.axis('off')
    # plt.savefig(os.path.join('redundancy_map.pdf'), bbox_inches='tight', dpi = 300)
    # plt.close()

    idx_img = 1 / idx_img
    idx_img = idx_img / sx / sx
    # line order
    idx = 0
    for i in th:
        for j in tw:
            norm_vec[idx] = idx_img[i:i + sx, j:j + sx].sum()
            idx += 1
    return norm_vec


def recover_countmap(pred, image, patch_sz, stride):
    pred = pred.reshape(-1)
    imH, imW = image.shape[2:4]
    cntMap = np.zeros((imH, imW), dtype=float)
    norMap = np.zeros((imH, imW), dtype=float)

    H = np.arange(0, imH - patch_sz + 1, stride)
    W = np.arange(0, imW - patch_sz + 1, stride)
    cnt = 0
    for h in H:
        for w in W:
            pixel_cnt = pred[cnt] / patch_sz / patch_sz
            cntMap[h:h + patch_sz, w:w + patch_sz] += pixel_cnt
            norMap[h:h + patch_sz, w:w + patch_sz] += np.ones((patch_sz, patch_sz))
            cnt += 1
    return cntMap / (norMap + 1e-12)
