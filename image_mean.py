import os
import numpy as np
from PIL import Image

img_dir = r"dataset_blueberry_augmented/image"
exts = {".jpg", ".jpeg", ".png"}  # 你主要是 .jpg

sum_c = np.zeros(3, dtype=np.float64)
sum_sq_c = np.zeros(3, dtype=np.float64)
num_pixels = 0

for name in os.listdir(img_dir):
    if os.path.splitext(name)[1].lower() not in exts:
        continue
    img = Image.open(os.path.join(img_dir, name)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
    # HWC -> 通道求和
    sum_c += arr.reshape(-1, 3).sum(axis=0)
    sum_sq_c += (arr.reshape(-1, 3) ** 2).sum(axis=0)
    num_pixels += arr.shape[0] * arr.shape[1]

mean = (sum_c / num_pixels).tolist()
var = (sum_sq_c / num_pixels - np.square(mean)).tolist()
std = np.sqrt(var).tolist()

print("mean:", mean)
print("std :", std)
