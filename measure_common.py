"""FLOPs / FPS 脚本的公共配置与模型加载。"""

import os

import numpy as np
import torch
import torch.nn.functional as F

image_scale = 1.0 / 255
image_mean = np.array([0.3751, 0.4294, 0.3128]).reshape((1, 1, 3))
image_std = np.array([1, 1, 1]).reshape((1, 1, 3))
output_stride = 8

DATA_DIR = "dataset_blueberry_augmented"
TEST_LIST = "final_test.txt"
MODEL_NAME = "IntegrateNet"
MODEL_PATH = "results/models/IntegrateNet/model_best.pth"
MODEL_TYPE = "integratenet"

CROP_H = 256
CROP_W = 256

# 验证/测试：等比缩放 + 填充到固定画布（letterbox）
VAL_INFER_H = 256
VAL_INFER_W = 256

MODEL_PRESETS = {
    "integratenet": {
        "name": "IntegrateNet",
        "path": "results/models/IntegrateNet/model_best.pth",
        "type": "integratenet",
    },
    "integratenet_enhanced_conv": {
        "name": "IntegrateNet_EnhancedConv",
        "path": "results/models/IntegrateNet_EnhancedConv/model_best.pth",
        "type": "integratenet_enhanced_conv",
    },
    "integratenet_ds": {
        "name": "IntegrateNet_DS",
        "path": "results/models/IntegrateNet_DS/model_best.pth",
        "type": "integratenet_ds",
    },
    "integratenet_ds_seg_guidance": {
        "name": "IntegrateNet_DS_SegGuidance",
        "path": "results/models/IntegrateNet_DS_SegGuidance/model_best.pth",
        "type": "integratenet_ds_seg_guidance",
    },
    "integratenet_seg_logits_guidance": {
        "name": "IntegrateNet_SegLogitsGuidance",
        "path": "results/models/IntegrateNet_SegLogitsGuidance/model_best.pth",
        "type": "integratenet_seg_logits_guidance",
    },
    "integratenet_enhanced_conv_ds": {
        "name": "IntegrateNet_EnhancedConv_DS",
        "path": "results/models/IntegrateNet_EnhancedConv_DS/model_best.pth",
        "type": "integratenet_enhanced_conv_ds",
    },
    "integratenet_enhanced_conv_seg_logits_guidance": {
        "name": "IntegrateNet_EnhancedConv_SegLogitsGuidance",
        "path": "results/models/IntegrateNet_EnhancedConv_SegLogitsGuidance/model_best.pth",
        "type": "integratenet_enhanced_conv_seg_logits_guidance",
    },
    "integratenet_enhanced_conv_ds_seg_guidance": {
        "name": "IntegrateNet_EnhancedConv_DS_SegGuidance",
        "path": "results/models/IntegrateNet_EnhancedConv_DS_SegGuidance/model_best.pth",
        "type": "integratenet_enhanced_conv_ds_seg_guidance",
    },
    "v3lite": {
        "name": "TasselNetv3_lite",
        "path": "results/models/v3lite/model_best.pth.tar",
        "type": "v3lite",
    },
    "v3seg": {
        "name": "TasselNetv3_seg",
        "path": "results/models/v3seg/model_best.pth.tar",
        "type": "v3seg",
    },
    "csrnet_seg": {
        "name": "CSRNet_Seg",
        "path": "results/models/CSRNet_Seg/model_best.pth",
        "type": "csrnet_seg",
    },
    "dmcount": {
        "name": "DM-Count",
        "path": "results/models/DMCount/model_best.pth",
        "type": "dmcount",
    },
    "p2pnet": {
        "name": "P2PNetWithSeg",
        "path": "results/models/P2PNet/model_best.pth",
        "type": "p2pnet",
    },
}

P2P_SW_MAX_SIZE = 3200
P2P_SW_STRIDE = 512
P2P_SW_CROP = 768


def _strip_module_prefix(state):
    if any(k.startswith("module.") for k in state):
        return {k[7:]: v for k, v in state.items()}
    return state


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


def _ensure_p2p_path():
    import sys
    root = os.path.dirname(os.path.abspath(__file__))
    p2p_dir = os.path.join(root, "CrowdCounting-P2PNet-main")
    for p in (root, p2p_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root, p2p_dir


def load_model(path=MODEL_PATH, model_type=MODEL_TYPE):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = _strip_module_prefix(_extract_state_dict(ckpt))

    if model_type == "v3lite":
        from V3liteNet import V3lite

        net = V3lite()
    elif model_type == "v3seg":
        from V3seg_net import V3seg

        net = V3seg()
    elif model_type == "csrnet_seg":
        from CSRNet_Seg import CSRNet_Seg

        net = CSRNet_Seg(load_weights=False)
    elif model_type == "integratenet_enhanced_conv":
        from IntegrateNet_EnhancedConv import IntegrateNet

        net = IntegrateNet()
    elif model_type == "integratenet_ds":
        from IntegrateNet_DS import IntegrateNet

        net = IntegrateNet()
    elif model_type == "integratenet_ds_seg_guidance":
        from IntegrateNet_DS_SegGuidance import IntegrateNet

        net = IntegrateNet(use_deep_supervision=True, use_seg_guidance=True)
    elif model_type == "integratenet_seg_logits_guidance":
        from IntegrateNet_SegLogitsGuidance import IntegrateNet

        net = IntegrateNet(use_seg_guidance=True)
    elif model_type == "integratenet_enhanced_conv_ds":
        from IntegrateNet_EnhancedConv_DS import IntegrateNet

        net = IntegrateNet(use_deep_supervision=True)
    elif model_type == "integratenet_enhanced_conv_seg_logits_guidance":
        from IntegrateNet_EnhancedConv_SegLogitsGuidance import IntegrateNet

        net = IntegrateNet(use_seg_guidance=True)
    elif model_type == "integratenet_enhanced_conv_ds_seg_guidance":
        from IntegrateNet_EnhancedConv_DS_SegGuidance import IntegrateNet

        net = IntegrateNet(use_deep_supervision=True, use_seg_guidance=True)
    elif model_type == "dmcount":
        import sys
        dm_dir = os.path.join(os.path.dirname(__file__), 'DM-Count-master')
        if dm_dir not in sys.path:
            sys.path.insert(0, dm_dir)
        from models import vgg19
        
        # DM-Count 模型结构（traval_dmcount.py 中的 SegHead + VGGWithSeg）
        import torch.nn as nn
        import torch.nn.functional as F
        
        class SegHead(nn.Module):
            def __init__(self, in_channels=128):
                super().__init__()
                self.head = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, kernel_size=1),
                )
            def forward(self, x):
                return self.head(x)
        
        class VGGWithSeg(nn.Module):
            def __init__(self, base_vgg, seg_head):
                super().__init__()
                self.features      = base_vgg.features
                self.reg_layer     = base_vgg.reg_layer
                self.density_layer = base_vgg.density_layer
                self.seg_head      = seg_head
            
            def forward(self, x):
                x = self.features(x)
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                feat = self.reg_layer(x)
                mu   = self.density_layer(feat).clamp(min=0)
                B, C, H, W = mu.size()
                mu_sum    = mu.view(B, -1).sum(1).view(B, 1, 1, 1).clamp(min=1e-6)
                mu_normed = mu / mu_sum
                seg_logits = self.seg_head(feat)
                return mu, mu_normed, seg_logits
        
        base_vgg = vgg19()
        seg_head = SegHead(in_channels=128)
        net = VGGWithSeg(base_vgg, seg_head)
    elif model_type == "p2pnet":
        import argparse

        _ensure_p2p_path()
        from models.backbone import build_backbone
        from traval_p2pnet import P2PNetWithSeg

        bb_args = argparse.Namespace(backbone="vgg16_bn")
        net = P2PNetWithSeg(build_backbone(bb_args), row=2, line=2)
    else:
        from IntegrateNet import IntegrateNet

        net = IntegrateNet()

    net.load_state_dict(state, strict=False)
    net.cuda().eval()
    return net


def apply_model_preset(preset_key):
    """设置全局 MODEL_NAME / MODEL_PATH / MODEL_TYPE，并返回三元组。"""
    global MODEL_NAME, MODEL_PATH, MODEL_TYPE
    if preset_key not in MODEL_PRESETS:
        raise ValueError(f"未知 preset: {preset_key}，可选: {list(MODEL_PRESETS)}")
    p = MODEL_PRESETS[preset_key]
    MODEL_NAME = p["name"]
    MODEL_PATH = p["path"]
    MODEL_TYPE = p["type"]
    return MODEL_NAME, MODEL_PATH, MODEL_TYPE


def resolve_model_config(model_preset=None, checkpoint=None):
    """解析模型配置；避免调用方 import 时缓存旧的全局变量。"""
    if model_preset:
        name, path, mtype = apply_model_preset(model_preset)
    else:
        name, path, mtype = MODEL_NAME, MODEL_PATH, MODEL_TYPE
    if checkpoint:
        path = checkpoint
    return name, path, mtype


def make_val_dataset(data_list=TEST_LIST, model_type=MODEL_TYPE):
    """验证集；P2PNet 使用 BlueberryP2PNet，其余使用 BlueberryDataset。"""
    if model_type == "p2pnet":
        return _make_p2pnet_val_dataset(data_list=data_list)
    from torchvision import transforms
    from Netdataset import BlueberryDataset, BBNormalize, BBToTensor, BBZeroPadding

    val_transforms = transforms.Compose([
        BBNormalize(scale=image_scale, std=image_std, mean=image_mean, train=False),
        BBToTensor(train=False),
        BBZeroPadding(output_stride, train=False),
    ])
    return BlueberryDataset(
        data_dir=DATA_DIR,
        data_list=data_list,
        train=False,
        transform=val_transforms,
    )


def _make_p2pnet_val_dataset(data_list=TEST_LIST, crop_size=256):
    """P2PNet 验证集包装：返回 {'image', 'filename'}。"""
    from torch.utils.data import Dataset

    _ensure_p2p_path()
    from crowd_datasets.blueberry.blueberry import BlueberryP2PNet

    class P2PNetValDataset(Dataset):
        def __init__(self):
            self.ds = BlueberryP2PNet(
                data_dir=DATA_DIR,
                data_list=data_list,
                crop_size=crop_size,
                num_patch=4,
                train=False,
                flip=False,
            )

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img_t, targets = self.ds[idx]
            tgt = targets[0]
            fname = tgt.get("filename", f"sample_{idx:03d}")
            return {"image": img_t, "filename": fname}

    return P2PNetValDataset()


def letterbox_params(h, w, box_h=VAL_INFER_H, box_w=VAL_INFER_W):
    """等比缩放至可放入 box，再居中填充；返回缩放后尺寸与 padding。"""
    scale = min(box_h / h, box_w / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    pad_h = box_h - nh
    pad_w = box_w - nw
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return {
        "orig_h": h,
        "orig_w": w,
        "box_h": box_h,
        "box_w": box_w,
        "scaled_h": nh,
        "scaled_w": nw,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "pad_bottom": pad_h - pad_top,
        "pad_right": pad_w - pad_left,
    }


def letterbox_input(input_tensor, box_h=VAL_INFER_H, box_w=VAL_INFER_W, pad_value=0.0):
    """
    等比缩放 + 居中填充到 box_h×box_w。
    返回 (tensor, need_restore, meta)。
    """
    _, _, h, w = input_tensor.shape
    if h == box_h and w == box_w:
        return input_tensor, False, None

    meta = letterbox_params(h, w, box_h, box_w)
    scaled = F.interpolate(
        input_tensor,
        size=(meta["scaled_h"], meta["scaled_w"]),
        mode="bilinear",
        align_corners=False,
    )
    padded = F.pad(
        scaled,
        (meta["pad_left"], meta["pad_right"], meta["pad_top"], meta["pad_bottom"]),
        mode="constant",
        value=pad_value,
    )
    return padded, True, meta


def restore_from_letterbox(tensor, meta, preserve_sum=True):
    """从 letterbox 画布裁出有效区域并上采样回原图尺寸。"""
    if meta is None:
        return tensor
    pt, pl = meta["pad_top"], meta["pad_left"]
    nh, nw = meta["scaled_h"], meta["scaled_w"]
    oh, ow = meta["orig_h"], meta["orig_w"]
    content = tensor[:, :, pt : pt + nh, pl : pl + nw]
    if (nh, nw) == (oh, ow):
        out = content
    else:
        out = F.interpolate(content, size=(oh, ow), mode="bilinear", align_corners=False)
    if preserve_sum:
        out = out * (oh * ow) / (nh * nw)
    return out


def inference_validate(net, input_tensor, return_seg=False):
    """
    验证推理：letterbox 到 640×640 → 前向 → 密度/分割还原到原图尺寸。
    """
    inp, need_restore, meta = letterbox_input(input_tensor)
    dic = net(inp)
    output = dic["density"]
    local_count = dic.get("local_count")
    seg_logits = dic.get("segmentation") if return_seg else None

    if need_restore:
        output = restore_from_letterbox(output, meta)
        if seg_logits is not None:
            seg_logits = restore_from_letterbox(seg_logits, meta, preserve_sum=False)
        local_count = None
    return output, local_count, seg_logits


def _upsample_density_preserve_sum(density, out_h, out_w):
    """1/8 密度图上采样到目标尺寸并保持总和（与 CSRNet_train 一致）。"""
    _, _, dh, dw = density.shape
    if (dh, dw) == (out_h, out_w):
        return density
    out = F.interpolate(density, size=(out_h, out_w), mode="bilinear", align_corners=False)
    out = out * (out_h * out_w) / (dh * dw)
    return out


def csrnet_forward(net, input_tensor, *, infer_size=None):
    """
    CSRNet_Seg 前向，返回密度图。
    infer_size: letterbox 到 N×N 后推理，并上采样密度到 N×N。
    无 infer_size: 整图推理，密度上采样回原图尺寸。
    """
    if infer_size is not None:
        if isinstance(infer_size, int):
            box_h = box_w = infer_size
        else:
            box_h, box_w = infer_size
        inp, _, _ = letterbox_input(input_tensor, box_h, box_w)
        density = net(inp)["density"]
        return _upsample_density_preserve_sum(density, box_h, box_w)
    _, _, h, w = input_tensor.shape
    density = net(input_tensor)["density"]
    return _upsample_density_preserve_sum(density, h, w)


def v3seg_forward(net, input_tensor, *, infer_size=None):
    """V3seg 前向，返回密度图 R（与 V3segtraval.py validate 一致）。"""
    if infer_size is not None:
        if isinstance(infer_size, int):
            box_h = box_w = infer_size
        else:
            box_h, box_w = infer_size
        inp, _, _ = letterbox_input(input_tensor, box_h, box_w)
        return net(inp, is_normalize=False)["R"]
    return net(input_tensor, is_normalize=False)["R"]


def v3lite_forward(net, input_tensor, *, infer_size=None):
    """
    V3lite 前向，返回密度图 R。
    infer_size: (H, W) 或 int；设置则 letterbox 到固定画布（与 IntegrateNet FPS 对齐）。
    """
    if infer_size is not None:
        if isinstance(infer_size, int):
            box_h = box_w = infer_size
        else:
            box_h, box_w = infer_size
        inp, _, _ = letterbox_input(input_tensor, box_h, box_w)
        return net(inp, is_normalize=False)["R"]
    return net(input_tensor, is_normalize=False)["R"]


def dmcount_forward(net, input_tensor):
    """DM-Count 前向，返回密度图 mu（已上采样到原图尺寸）。"""
    _, _, h, w = input_tensor.shape
    mu, mu_normed, seg_logits = net(input_tensor)  # 三元组输出
    # mu 是 1/8 分辨率，上采样到原图
    return _upsample_density_preserve_sum(mu, h, w)


def p2pnet_forward(
    net,
    input_tensor,
    max_size=P2P_SW_MAX_SIZE,
    stride=P2P_SW_STRIDE,
    crop_size=P2P_SW_CROP,
):
    """P2PNet 全图推理（与 traval_p2pnet.validate 一致，大图 sliding window）。"""
    img = input_tensor if input_tensor.dim() == 4 else input_tensor.unsqueeze(0)
    _, _, h, w = img.shape

    if h <= max_size and w <= max_size:
        return net(img)["seg_logits"]

    seg_logits_full = torch.zeros((1, 1, h, w), device=img.device)
    count_map = torch.zeros((1, 1, h, w), device=img.device)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + crop_size, h)
            x_end = min(x + crop_size, w)
            y_start = max(0, y_end - crop_size)
            x_start = max(0, x_end - crop_size)

            patch = img[:, :, y_start:y_end, x_start:x_end]
            seg_p = net(patch)["seg_logits"]
            seg_logits_full[:, :, y_start:y_end, x_start:x_end] += seg_p
            count_map[:, :, y_start:y_end, x_start:x_end] += 1.0

    return seg_logits_full / count_map.clamp(min=1.0)


def validate_forward(net, input_tensor, model_type=MODEL_TYPE, infer_size=None):
    """完整部署前向。"""
    if model_type == "v3lite":
        return v3lite_forward(net, input_tensor, infer_size=infer_size)
    if model_type == "v3seg":
        return v3seg_forward(net, input_tensor, infer_size=infer_size)
    if model_type == "csrnet_seg":
        return csrnet_forward(net, input_tensor, infer_size=infer_size)
    if model_type == "dmcount":
        return dmcount_forward(net, input_tensor)
    if model_type == "p2pnet":
        return p2pnet_forward(net, input_tensor)
    if model_type in (
        "integratenet",
        "integratenet_enhanced_conv",
        "integratenet_ds",
        "integratenet_ds_seg_guidance",
        "integratenet_seg_logits_guidance",
        "integratenet_enhanced_conv_ds",
        "integratenet_enhanced_conv_seg_logits_guidance",
        "integratenet_enhanced_conv_ds_seg_guidance",
    ):
        output, _, _ = inference_validate(net, input_tensor, return_seg=False)
        return output


def letterbox_desc(h, w):
    """用于日志：描述 letterbox 后的有效区域。"""
    if h == VAL_INFER_H and w == VAL_INFER_W:
        return f"{h}×{w}"
    m = letterbox_params(h, w)
    return (
        f"{h}×{w} → 缩放 {m['scaled_h']}×{m['scaled_w']} "
        f"+ 填充至 {VAL_INFER_H}×{VAL_INFER_W}"
    )
