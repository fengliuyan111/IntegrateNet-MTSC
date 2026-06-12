"""
测量 IntegrateNet 推理 FPS（GPU，不含 DataLoader I/O）。

计时: letterbox + net + 密度图还原到原图（完整部署流程）

用法:
  python measure_fps.py --model v3lite
  python measure_fps.py --model v3lite --infer-size 256   # 与 IntegrateNet 对齐
  python measure_fps.py --model integratenet --repeat 30
  python measure_fps.py --model dmcount
  python measure_fps.py --model p2pnet
"""

import argparse
import gc
import statistics
import time

import torch
from torch.utils.data import DataLoader

from measure_common import (
    TEST_LIST,
    MODEL_PRESETS,
    letterbox_desc,
    load_model,
    make_val_dataset,
    resolve_model_config,
    validate_forward,
)

DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 30


def _run_timed(forward_fn, repeat):
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times) * 1000


def _summarize(per_image_ms):
    if not per_image_ms:
        return None
    mean_ms = statistics.mean(per_image_ms)
    std_ms = statistics.stdev(per_image_ms) if len(per_image_ms) > 1 else 0.0
    return {
        "n": len(per_image_ms),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "fps": 1000.0 / mean_ms,
    }


def measure_fps(net, val_loader, *, warmup, repeat, clear_cache, model_type, infer_size=None):
    per_image = []
    n_total = len(val_loader) - warmup

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            fname_raw = sample.get("filename", "")
            if isinstance(fname_raw, (list, tuple)):
                fname = fname_raw[0] if fname_raw else ""
            else:
                fname = fname_raw or ""
            img = None
            try:
                img = sample["image"].cuda()
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                _, _, h, w = img.shape

                if i < warmup:
                    validate_forward(
                        net, img, model_type=model_type, infer_size=infer_size
                    )
                    torch.cuda.synchronize()
                    continue

                ms = _run_timed(
                    lambda: validate_forward(
                        net, img, model_type=model_type, infer_size=infer_size
                    ),
                    repeat,
                )
                per_image.append(ms)

                n = len(per_image)
                size_desc = f"{h}×{w}" if model_type == "p2pnet" else letterbox_desc(h, w)
                print(f"  [{n}/{n_total}] {fname}  {size_desc}  {ms:.4f}ms")

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "alloc_failed" in str(e).lower():
                    print(f"  [跳过] {fname}  显存不足: {e}")
                    if clear_cache:
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    raise
            finally:
                del img
                if clear_cache:
                    torch.cuda.empty_cache()

    return _summarize(per_image)


def print_summary(result, repeat, model_type, infer_size=None):
    if result is None:
        return

    print("\n--- [pipeline] ---")
    if model_type in ("v3lite", "v3seg", "csrnet_seg"):
        if infer_size is not None:
            sz = infer_size if isinstance(infer_size, int) else infer_size
            if model_type == "csrnet_seg":
                print(f"计时: letterbox 至 {sz}×{sz} + net + 密度上采样至 {sz}×{sz}")
            else:
                print(f"计时: letterbox 至 {sz}×{sz} + net(x) → R（与 IntegrateNet 画布对齐）")
        else:
            labels = {"v3lite": "V3lite", "v3seg": "V3seg", "csrnet_seg": "CSRNet_Seg"}
            print(f"计时: 整图 net + 密度上采样（与 {labels[model_type]} validate 一致）")
    elif model_type == "p2pnet":
        print("计时: 全图 forward（>3200px 启用 sliding window，与 traval_p2pnet 一致）")
    elif model_type == "dmcount":
        print("计时: 全图 forward + 密度上采样（与 traval_dmcount 一致）")
    else:
        print("计时: letterbox + net + 密度图还原到原图")

    print(f"每张图 repeat={repeat}，图像级 mean ± std")
    print(f"成功: {result['n']} 张")
    print(f"延迟: {result['mean_ms']:.4f} ± {result['std_ms']:.4f} ms/image")
    print(f"FPS:  {result['fps']:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="模型推理 FPS benchmark")
    p.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="每张图后 empty_cache（默认关闭，避免计时变慢）",
    )
    p.add_argument(
        "--model",
        choices=tuple(MODEL_PRESETS.keys()),
        default=None,
        help="模型预设: integratenet | ... | csrnet_seg | dmcount | p2pnet",
    )
    p.add_argument("--checkpoint", default=None, help="权重路径（覆盖 --model 默认路径）")
    p.add_argument(
        "--infer-size",
        type=int,
        default=None,
        metavar="N",
        help="v3lite/v3seg/csrnet_seg: letterbox 到 N×N；IntegrateNet 忽略此项",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name, model_path, model_type = resolve_model_config(
        model_preset=args.model, checkpoint=args.checkpoint
    )

    torch.backends.cudnn.benchmark = True

    valset = make_val_dataset(model_type=model_type)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    net = load_model(model_path, model_type=model_type)

    print("=" * 56)
    print(f"模型: {model_name}")
    print(f"权重: {model_path}")
    print(f"数据: {TEST_LIST}  共 {len(valset)} 张")
    print(f"预热 {args.warmup} 张，计时 {len(valset) - args.warmup} 张，repeat={args.repeat}")
    infer_size = args.infer_size
    if model_type in ("v3lite", "v3seg", "csrnet_seg"):
        if infer_size:
            if model_type == "csrnet_seg":
                print(f"计时: letterbox {infer_size}×{infer_size} + net + 密度上采样")
            else:
                print(f"计时: letterbox 至 {infer_size}×{infer_size} + net(x) → R")
        else:
            print("计时: 整图 net + 密度上采样（无 letterbox）")
    elif model_type == "p2pnet":
        print("计时: 全图 forward（>3200px 启用 sliding window）")
    else:
        print("计时: letterbox + net + 密度图还原到原图")

    if torch.cuda.is_available():
        print(f"设备: {torch.cuda.get_device_name(0)}")
    print("=" * 56)

    result = measure_fps(
        net,
        val_loader,
        warmup=args.warmup,
        repeat=args.repeat,
        clear_cache=args.clear_cache,
        model_type=model_type,
        infer_size=infer_size,
    )
    print_summary(result, args.repeat, model_type=model_type, infer_size=infer_size)
