import os
import json
import random
import copy
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


class DataAugmentation:
    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.image_dir = os.path.join(source_dir, 'images')
        self.dot_dir = os.path.join(source_dir, 'dot')
        self.land_dir = os.path.join(source_dir, 'land1')

        # 创建输出目录
        self.aug_image_dir = os.path.join(output_dir, 'images')
        self.aug_dot_dir = os.path.join(output_dir, 'dot')
        self.aug_mask_dir = os.path.join(output_dir, 'mask')
        os.makedirs(self.aug_image_dir, exist_ok=True)
        os.makedirs(self.aug_dot_dir, exist_ok=True)
        os.makedirs(self.aug_mask_dir, exist_ok=True)

        # 定义数据增强pipeline
        self.transforms = [
            A.Compose([A.Rotate(limit=(85, 95), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
            A.Compose([A.Rotate(limit=(90, 90), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
            A.Compose([A.Rotate(limit=(95, 105), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
            A.Compose([A.RandomScale(scale_limit=(-0.3, -0.2), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
            A.Compose([A.RandomScale(scale_limit=(-0.2, -0.1), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
            A.Compose([A.RandomScale(scale_limit=(0.1, 0.2), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.2,0.3), contrast_limit=(0.2,0.3), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.3,-0.2), contrast_limit=(0.2,0.3), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.2,0.3), contrast_limit=(-0.3,-0.2), p=1.0)],
                      keypoint_params=A.KeypointParams(format='xy')),
        ]

    def load_annotations(self, dot_path, land_path):
        with open(dot_path, 'r', encoding='utf-8') as f:
            dot_data = json.load(f)
        with open(land_path, 'r', encoding='utf-8') as f:
            land_data = json.load(f)
        return dot_data, land_data

    def save_annotations(self, dot_data, dot_path):
        with open(dot_path, 'w') as f:
            json.dump(dot_data, f, indent=2)

    def polygons_to_mask(self, polygons, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def mask_to_polygons(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for cnt in contours:
            if cnt.shape[0] > 2:  # 至少3个点
                poly = cnt[:, 0, :].tolist()
                polys.append(poly)
        return polys

    def apply_augmentation(self, image, dot_data, land_data, transform):
        # --- 准备 keypoints ---
        keypoints = []
        for shape in dot_data['shapes']:
            if shape['shape_type'] == "point":
                keypoints.append(shape['points'][0])

        # --- 准备 mask ---
        polygons = [shape['points'] for shape in land_data['shapes'] if shape['shape_type'] == "polygon"]
        h, w = image.shape[:2]
        mask = self.polygons_to_mask(polygons, h, w)

        # --- 应用增强 ---
        augmented = transform(image=image, keypoints=keypoints, mask=mask)
        aug_image = augmented['image']
        aug_keypoints = augmented['keypoints']
        aug_mask = augmented['mask']

        # --- 重建 dot_data ---
        new_dot_data = copy.deepcopy(dot_data)
        new_dot_data['imageHeight'] = aug_image.shape[0]
        new_dot_data['imageWidth'] = aug_image.shape[1]
        new_dot_data['shapes'] = [{"label": "point", "points":[kp], "shape_type":"point", "flags":{}} for kp in aug_keypoints]

        return aug_image, new_dot_data, aug_mask

    def process_dataset(self):
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        augmented_files = []

        for img_file in tqdm(image_files):
            img_path = os.path.join(self.image_dir, img_file)
            dot_path = os.path.join(self.dot_dir, img_file.replace('.jpg','.json'))
            land_path = os.path.join(self.land_dir, img_file.replace('.jpg','.json'))

            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            dot_data, land_data = self.load_annotations(dot_path, land_path)

            base_name = os.path.splitext(img_file)[0]

            # ✅ 保存原始图像和标签
            cv2.imwrite(os.path.join(self.aug_image_dir, f"{base_name}_orig.jpg"),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # 保存原始点标注
            self.save_annotations(dot_data, os.path.join(self.aug_dot_dir, f"{base_name}_orig.json"))
            # 保存原始 mask
            polygons = [shape['points'] for shape in land_data['shapes'] if shape['shape_type'] == "polygon"]
            h, w = image.shape[:2]
            orig_mask = self.polygons_to_mask(polygons, h, w)
            cv2.imwrite(os.path.join(self.aug_mask_dir, f"{base_name}_orig.png"), orig_mask)

            augmented_files.append(f"{base_name}_orig.jpg")

            # ✅ 接下来才是增强部分
            for i, transform in enumerate(self.transforms):
                aug_type = i // 3
                aug_variant = i % 3
                new_name = f"{base_name}_aug_{aug_type}_{aug_variant}"

                aug_image, aug_dot_data, aug_mask = self.apply_augmentation(
                    image.copy(), dot_data.copy(), land_data.copy(), transform
                )
                # 保存增强后的图像
                cv2.imwrite(os.path.join(self.aug_image_dir, f"{new_name}.jpg"),
                            cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                # 保存增强后的点标注 JSON
                self.save_annotations(aug_dot_data, os.path.join(self.aug_dot_dir, f"{new_name}.json"))
                # 保存增强后的 mask
                mask_path = os.path.join(self.aug_mask_dir, f"{new_name}.png")
                cv2.imwrite(mask_path, aug_mask)

                augmented_files.append(f"{new_name}.jpg")
        return augmented_files

    def split_dataset(self, augmented_files, train_ratio=0.8):
        random.shuffle(augmented_files)
        split_idx = int(len(augmented_files)*train_ratio)
        train_files = augmented_files[:split_idx]
        test_files = augmented_files[split_idx:]

        with open(os.path.join(self.output_dir,'train.txt'),'w') as f:
            f.write('\n'.join(train_files))
        with open(os.path.join(self.output_dir,'test.txt'),'w') as f:
            f.write('\n'.join(test_files))


def main():
    source_dir = 'dataset_blueberry'
    output_dir = 'dataset_blueberry_augmented'
    augmentor = DataAugmentation(source_dir, output_dir)

    print("开始数据增强...")
    augmented_files = augmentor.process_dataset()

    print("分割数据集...")
    augmentor.split_dataset(augmented_files)

    print("数据处理完成！")
    print(f"增强后的数据保存在: {output_dir}")
    print(f"训练集和测试集的文件列表保存在: {output_dir}/train.txt 和 {output_dir}/test.txt")


if __name__ == "__main__":
    main()
