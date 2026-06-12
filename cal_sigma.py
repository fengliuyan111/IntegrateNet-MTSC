import os, numpy as np
from scipy.spatial import cKDTree

dot_dir = r"dataset_blueberry_augmented/dot_txt"
all_dist = []

for fn in os.listdir(dot_dir):
    if not fn.endswith(".txt"): continue
    pts = []
    with open(os.path.join(dot_dir, fn), "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            x, y = map(float, s.split(","))
            pts.append([x, y])
    if len(pts) < 2: continue
    P = np.array(pts, dtype=np.float32)
    tree = cKDTree(P)
    # 取每点到最近的3个邻居（去掉自己），平均距离作为该点的局部尺度
    dists, _ = tree.query(P, k=min(4, len(P)))  # 返回 [self, n1, n2, n3]
    local = dists[:, 1:4].mean(axis=1) if dists.shape[1] > 1 else dists[:, 1]
    all_dist.append(local)

if len(all_dist) == 0:
    print("No enough points found.")
else:
    d = np.concatenate(all_dist)
    med = float(np.median(d))
    p25, p75 = np.percentile(d, [25, 75])
    # 建议固定 sigma 与自适应 β
    sigma_fixed = 0.3 * med
    beta = 0.3
    print(f"Nearest-neighbor distance: median={med:.1f}px, P25={p25:.1f}, P75={p75:.1f}")
    print(f"Suggested fixed sigma ≈ {sigma_fixed:.1f} px (range ~{0.2*med:.1f}-{0.4*med:.1f})")
    print(f"Suggested adaptive: sigma_i = {beta} * mean(d1..d3), clip e.g. [4, 40] px")
