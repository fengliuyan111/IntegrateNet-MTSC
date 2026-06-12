"""
测量模型 FLOPs，提供两种方式：

  thop        MACs×2，与 YOLOv8 同口径；不计 Sigmoid/Pool/ReLU
              → 适合论文中跨模型横向对比（口径统一）

  torch_profiler  统计实际执行的所有算子 FLOPs（含 Sigmoid/Pool/BN/等）
              → 更能反映 CBAM 等注意力模块的真实算力开销

用法:
  python measure_flops.py                  # 两种都跑
  python measure_flops.py --method thop
  python measure_flops.py --method profiler
  python measure_flops.py --model v3lite   # 额外输出【V3lite 完整 FLOPs】
  python measure_flops.py --model v3seg    # 额外输出【V3seg 完整 FLOPs】
  python measure_flops.py --model csrnet_seg
  python measure_flops.py --model dmcount
  python measure_flops.py --model p2pnet
  python measure_v3lite_flops.py           # 仅 V3lite 完整 FLOPs
  python measure_v3seg_flops.py            # 仅 V3seg 完整 FLOPs
"""

import argparse
import copy
import gc

import torch

from measure_common import (
    CROP_H,
    CROP_W,
    MODEL_PRESETS,
    load_model,
    output_stride,
    resolve_model_config,
)

INPUT_H = CROP_H
INPUT_W = CROP_W


def padded_hw(h, w, stride=output_stride):
    ph = (stride - h % stride) % stride
    pw = (stride - w % stride) % stride
    return h + ph, w + pw


def count_params(net):
    return sum(p.numel() for p in net.parameters())


# ─── Method 1: thop ──────────────────────────────────────────────────────────

def profile_thop(net, h, w, device="cuda"):
    """MACs×2 / 1e9，与 YOLOv8 同口径。不计 Sigmoid / Pool / ReLU。"""
    from thop import profile

    H, W = padded_hw(h, w)
    net_p = copy.deepcopy(net).to(device).eval()
    dummy = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        macs, _ = profile(net_p, inputs=(dummy,), verbose=False)
    del net_p, dummy
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return macs * 2 / 1e9, H, W


def profile_thop_breakdown(net, h, w, device="cuda"):
    """按顶层子模块分解 MACs。"""
    from thop import profile

    H, W = padded_hw(h, w)
    net_p = copy.deepcopy(net).to(device).eval()
    dummy = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        macs, _, layer_info = profile(
            net_p, inputs=(dummy,), verbose=False, ret_layer_info=True
        )
    del net_p, dummy
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    parts = sorted(
        [(name, ops, prm) for name, (ops, prm, _) in layer_info.items()],
        key=lambda x: x[1], reverse=True,
    )
    return macs * 2 / 1e9, H, W, parts


# ─── Method 2: torch.profiler ────────────────────────────────────────────────

def profile_torch_profiler(net, h, w, device="cuda", warmup=3):
    """
    使用 torch.profiler 统计所有算子的 FLOPs（含 Pool/Sigmoid/BN 等）。
    返回 GFLOPs（浮点操作数 / 1e9）。
    注意：部分自定义算子（如 CARAFE matmul）可能仍未被注册，结果仍为下界。
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("[profiler] CUDA 不可用，回退 CPU")
        device = "cpu"

    H, W = padded_hw(h, w)
    net_p = copy.deepcopy(net).to(device).eval()
    dummy = torch.randn(1, 3, H, W, device=device)

    # 预热，让 cuDNN 选好算法
    with torch.no_grad():
        for _ in range(warmup):
            net_p(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        with_flops=True,
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            net_p(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

    # 汇总所有事件的 flops
    total_flops = sum(
        e.flops for e in prof.key_averages() if e.flops > 0
    )

    # 按算子名称聚合（取 Top-10）
    op_flops = {}
    for e in prof.key_averages():
        if e.flops > 0:
            op_flops[e.key] = op_flops.get(e.key, 0) + e.flops
    top_ops = sorted(op_flops.items(), key=lambda x: x[1], reverse=True)[:10]

    del net_p, dummy
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return total_flops / 1e9, H, W, top_ops


# ─── 打印 ─────────────────────────────────────────────────────────────────────

def print_thop(net, h, w, device):
    gflops, H, W = profile_thop(net, h, w, device)
    print(f"\n【thop】输入 {H}×{W}")
    print(f"  GFLOPs = {gflops:.4f} G  （MACs×2，不含 Sigmoid/Pool/ReLU）")
    print(f"  ↑ 与 YOLOv8 同口径，适合跨模型公平对比")

    gflops_bd, _, _, parts = profile_thop_breakdown(net, h, w, device)
    total_g = gflops_bd
    print(f"\n  {'模块':<16} {'GFLOPs':>9} {'占比':>7}")
    print("  " + "-" * 34)
    for name, ops, _ in parts[:10]:
        pct = 100.0 * ops * 2 / 1e9 / total_g if total_g > 0 else 0
        print(f"  {name:<16} {ops*2/1e9:>9.4f} {pct:>6.1f}%")
    print("  " + "-" * 34)
    print(f"  {'合计':<16} {total_g:>9.4f}")
    return gflops


def print_profiler(net, h, w, device):
    gflops, H, W, top_ops = profile_torch_profiler(net, h, w, device)
    print(f"\n【torch.profiler】输入 {H}×{W}")
    print(f"  GFLOPs = {gflops:.4f} G  （含 Pool/Sigmoid/BN 等实际算子，仍为下界）")
    print(f"  ↑ 更能体现 CBAM/Attention 的真实算力开销")
    print(f"\n  {'算子':<30} {'GFLOPs':>9}")
    print("  " + "-" * 42)
    for name, flops in top_ops:
        print(f"  {name:<30} {flops/1e9:>9.4f}")
    print("  " + "-" * 42)
    print(f"  {'合计（注册算子）':<30} {gflops:>9.4f}")
    return gflops


# ─── 主程序 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        choices=("both", "thop", "profiler"),
        default="both",
        help="thop=MACs×2（YOLOv8 口径）; profiler=torch.profiler（含 Sigmoid/Pool）",
    )
    p.add_argument(
        "--model",
        choices=tuple(MODEL_PRESETS.keys()),
        default=None,
        help="模型预设: integratenet | ... | csrnet_seg | dmcount | p2pnet",
    )
    p.add_argument("--checkpoint", default=None, help="权重路径（覆盖 --model 默认路径）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name, model_path, model_type = resolve_model_config(
        model_preset=args.model, checkpoint=args.checkpoint
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = INPUT_H, INPUT_W

    net = load_model(model_path, model_type=model_type)
    params_m = count_params(net) / 1e6

    print("=" * 56)
    print(f"模型: {model_name}")
    print(f"权重: {model_path}")
    print(f"参数量: {params_m:.2f} M")
    print(f"输入: {h}×{w}  (eval mode)")
    if device == "cuda":
        print(f"设备: {torch.cuda.get_device_name(0)}")
    print("=" * 56)

    thop_g = profiler_g = None

    if args.method in ("both", "thop"):
        thop_g = print_thop(net, h, w, device)

    if args.method in ("both", "profiler"):
        profiler_g = print_profiler(net, h, w, device)

    if model_type == "v3lite":
        from measure_v3lite_flops import print_v3lite_complete

        print_v3lite_complete(net, h, w, device, profiler_g=profiler_g)
    elif model_type == "v3seg":
        from measure_v3seg_flops import print_v3seg_complete

        print_v3seg_complete(net, h, w, device, profiler_g=profiler_g)

    if thop_g is not None and profiler_g is not None:
        print(f"\n{'='*56}")
        print(f"对比汇总")
        print(f"  thop GFLOPs      : {thop_g:.4f} G  （论文标准口径）")
        print(f"  profiler GFLOPs  : {profiler_g:.4f} G  （含 Sigmoid/Pool 等）")
        diff = profiler_g - thop_g
        print(f"  差值             : {diff:+.4f} G  ← 注意力/激活等额外开销")
        print(
            "\n  结论: 若 CBAM 模型 profiler 差值明显大于 SegLogits 模型，"
            "说明 thop 低估了 CBAM 的真实算力，FPS 低于预期是合理的。"
        )
        print("=" * 56)
