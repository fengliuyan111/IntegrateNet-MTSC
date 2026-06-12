import os
import random
import json


def convert_dot_json_to_txt():
    """
    将dot文件夹中的JSON点标注文件转换为包含点坐标的TXT文件
    """
    source_dir = 'dataset_blueberry_augmented'
    dot_dir = os.path.join(source_dir, 'dot')
    output_dir = os.path.join(source_dir, 'dot_txt')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(dot_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        json_path = os.path.join(dot_dir, json_file)
        txt_file = json_file.replace('.json', '.txt')
        txt_path = os.path.join(output_dir, txt_file)
        
        try:
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取点坐标
            points = []
            for shape in data.get('shapes', []):
                if shape.get('shape_type') == 'point' and shape.get('points'):
                    # 获取点坐标
                    x, y = shape['points'][0]
                    points.append(f"{x:.2f},{y:.2f}")
            
            # 保存到TXT文件
            with open(txt_path, 'w', encoding='utf-8') as f:
                for point in points:
                    f.write(f"{point}\n")
            
            print(f"转换完成：{json_file} -> {txt_file} ({len(points)} 个点)")
            
        except Exception as e:
            print(f"转换失败 {json_file}: {str(e)}")
    
    print(f"所有点标注文件已转换完成，保存到：{output_dir}")


def main():
    print("=== 蓝莓数据集处理 ===")
    print()
    
    print("1. 将dot JSON文件转换为TXT格式...")
    convert_dot_json_to_txt()
    print()
    
    print("处理完成！")
    print("- 点标注TXT文件保存在：dataset_blueberry/dot_txt/")


if __name__ == "__main__":
    main()
