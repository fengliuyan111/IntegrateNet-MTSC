import os
import json
import numpy as np
import cv2


def labelme_to_mask(json_path, output_dir, img_dir=None, use_json_filename=False):
    """
    将Labelme标注的JSON文件转换为分割掩码

    Args:
        json_path: Labelme JSON文件路径
        output_dir: 输出掩码的目录
        img_dir: 图像所在目录（可选，默认与JSON文件同目录）
        use_json_filename: 是否使用JSON文件名作为图像文件名（默认False，使用JSON内的imagePath）
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图像文件名
    if use_json_filename:
        # 使用JSON文件名作为图像文件名（Image1.json -> Image1.xxx）
        json_basename = os.path.splitext(os.path.basename(json_path))[0]
        # 尝试多种图像格式
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        base_dir = img_dir if img_dir else os.path.dirname(json_path)
        
        img_path = None
        for ext in possible_extensions:
            test_path = os.path.join(base_dir, json_basename + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        # 如果没找到，默认使用.jpg
        if img_path is None:
            img_path = os.path.join(base_dir, json_basename + '.jpg')
    else:
        # 从JSON中获取图像文件名
        image_filename = data.get('imagePath', '')
        # 如果指定了图像目录，使用指定目录；否则使用JSON文件所在目录
        if img_dir:
            img_path = os.path.join(img_dir, os.path.basename(image_filename))
        else:
            json_dir_path = os.path.dirname(json_path)
            img_path = os.path.join(json_dir_path, image_filename)

    # 读取图像
    img = cv2.imread(img_path)
    if img is not None:
        print(f'Successfully loaded image from: {img_path}')
        # 获取图像尺寸
        height, width = img.shape[:2]
    else:
        print(f'Failed to load image from: {img_path}')
        # 如果无法读取图像，尝试从JSON中获取尺寸
        height = data.get('imageHeight', 512)
        width = data.get('imageWidth', 512)
        print(f'Using dimensions from JSON: {width}x{height}')

    # 创建掩码图像
    mask = np.zeros((height, width), dtype=np.uint8)

    # 处理每个标注对象
    for shape in data['shapes']:
        # 获取多边形点
        points = shape['points']
        # 转换为numpy数组
        points = np.array(points, dtype=np.int32)

        # 填充多边形区域
        cv2.fillPoly(mask, [points], 1)

    # 保存掩码
    output_name = os.path.splitext(os.path.basename(json_path))[0] + '.png'
    output_path = os.path.join(output_dir, output_name)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存掩码图像
    cv2.imwrite(output_path, mask * 255)
    print(f'Saved mask to {output_path}')

    return mask


def batch_convert(json_dir, output_dir, img_dir=None, use_json_filename=False):
    """
    批量转换目录下的所有JSON文件
    
    Args:
        json_dir: JSON文件所在目录
        output_dir: 输出掩码的目录
        img_dir: 图像所在目录（可选，默认与JSON文件同目录）
        use_json_filename: 是否使用JSON文件名作为图像文件名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有JSON文件
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            print(f'\nProcessing {filename}...')
            try:
                labelme_to_mask(json_path, output_dir, img_dir, use_json_filename)
            except Exception as e:
                print(f'Error processing {filename}: {str(e)}')


if __name__ == '__main__':
    # 请修改这些路径为您的实际路径
    json_dir = r"C:\Users\PC\Desktop\pipa\jsons"  # Labelme JSON文件目录
    output_dir = r"C:\Users\PC\Desktop\pipa\masks"  # 输出掩码目录
    img_dir = r"C:\Users\PC\Desktop\pipa\images"  # 原始图像目录
    
    # 设置为True: 使用JSON文件名作为图像文件名 (Image1.json -> Image1.jpg)
    # 设置为False: 使用JSON内部的imagePath字段
    use_json_filename = True

    print(f"Converting JSON files from: {json_dir}")
    print(f"Loading images from: {img_dir}")
    print(f"Saving masks to: {output_dir}")
    print(f"Use JSON filename as image name: {use_json_filename}")

    # 执行转换
    batch_convert(json_dir, output_dir, img_dir, use_json_filename)
