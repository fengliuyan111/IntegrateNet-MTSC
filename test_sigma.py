import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

img_dir = r"dataset_blueberry_augmented/image"
dot_dir = r"dataset_blueberry_augmented/dot_txt"

# 你要验证的几个sigma（建议：中值与上下界）
sigmas = [32.8, 50, 65.5]  # px


def load_points(txt_path):
    pts = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            x, y = map(float, s.split(","))
            pts.append((x, y))
    return pts


def make_density(h, w, points, sigma):
    dm = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        xx, yy = int(round(x)), int(round(y))
        if 0 <= yy < h and 0 <= xx < w:
            dm[yy, xx] = 1.0
    # 对“脉冲点图”做高斯滤波，等价于每点放置一个高斯核
    dm = gaussian_filter(dm, sigma=sigma, mode="constant")
    return dm


# 挑几张样例
samples = [name for name in os.listdir(img_dir) if name.lower().endswith(".jpg")][:3]

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 创建输出目录
output_dir = "sigma_test_results"
os.makedirs(output_dir, exist_ok=True)

for name in samples:
    stem, _ = os.path.splitext(name)
    img = Image.open(os.path.join(img_dir, name)).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    pts = load_points(os.path.join(dot_dir, stem + ".txt"))

    # 1. 保存原始图像
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr)
    ax.set_title("Original image", fontsize=48, fontweight='bold')
    ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{stem}_original.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {save_path}")

    # 2. 保存不同sigma的密度图
    for s in sigmas:
        fig, ax = plt.subplots(figsize=(8, 8))
        dm = make_density(h, w, pts, sigma=s)
        ax.imshow(arr)
        ax.imshow(dm, cmap="jet", alpha=0.45)  # 叠加热力图
        ax.set_title(f"σ = {s:.1f} px", fontsize=48, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{stem}_sigma_{s:.1f}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved: {save_path}")

print(f"\nAll images saved to: {output_dir}/")
