import os
import random
from PIL import Image


def random_crop_image(image_path, output_folder, crop_width=1280, crop_height=720, num_crops=3):
    """对单张图片随机裁剪"""
    img = Image.open(image_path)
    img_width, img_height = img.size

    if img_width < crop_width or img_height < crop_height:
        print(f"跳过 {image_path}: 图像太小，无法裁剪。")
        return

    img_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(num_crops):
        left = random.randint(0, img_width - crop_width)
        top = random.randint(0, img_height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        cropped_img = img.crop((left, top, right, bottom))

        save_name = f"{img_name}_crop_{i + 1}.jpg"
        save_path = os.path.join(output_folder, save_name)
        cropped_img.save(save_path)
        print(f"已保存: {save_path}")


def batch_crop_images(input_folder, output_folder, crop_width=1280, crop_height=720, num_crops=5):
    """批量裁剪文件夹内的所有图片"""
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        random_crop_image(image_path, output_folder, crop_width, crop_height, num_crops)


# 使用示例
input_folder = r"C:\Users\PC\Desktop\lanmei_drone\DJI_lanmei_3840_2160"  # 你的图片文件夹路径
output_folder = r"C:\Users\PC\Desktop\lanmei_drone\sample-1280-720"  # 输出文件夹路径
batch_crop_images(input_folder, output_folder)
