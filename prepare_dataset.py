import os
import random
import shutil
from pathlib import Path

def prepare_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, seed=42):
    """
    自动准备YOLO格式的数据集
    
    Args:
        images_dir (str): 原始图片目录路径
        labels_dir (str): 原始标签目录路径
        output_dir (str): 输出数据集目录路径
        train_ratio (float): 训练集比例，默认0.8
        seed (int): 随机种子，确保每次划分一致
    """
    
    # 设置随机种子以确保可重复性
    random.seed(seed)
    
    # 创建输出目录结构
    dirs_to_create = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'labels', 'train'),
        os.path.join(output_dir, 'labels', 'val')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 获取所有图片文件名（不带扩展名）
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    base_names = [os.path.splitext(f)[0] for f in image_files]
    
    print(f"找到 {len(base_names)} 张图片")
    
    # 随机打乱文件列表
    random.shuffle(base_names)
    
    # 计算训练集和验证集的分割点
    split_index = int(len(base_names) * train_ratio)
    train_files = base_names[:split_index]
    val_files = base_names[split_index:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 复制训练集文件
    copied_count = 0
    for base_name in train_files:
        # 查找对应的图片文件（考虑不同的扩展名）
        image_ext = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if os.path.exists(os.path.join(images_dir, base_name + ext)):
                image_ext = ext
                break
        
        if image_ext is None:
            print(f"警告: 未找到 {base_name} 的图片文件")
            continue
        
        # 复制图片文件
        src_image = os.path.join(images_dir, base_name + image_ext)
        dst_image = os.path.join(output_dir, 'images', 'train', base_name + image_ext)
        shutil.copy2(src_image, dst_image)
        
        # 复制标签文件
        src_label = os.path.join(labels_dir, base_name + '.txt')
        dst_label = os.path.join(output_dir, 'labels', 'train', base_name + '.txt')
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
            copied_count += 1
        else:
            print(f"警告: 未找到标签文件 {src_label}")
    
    print(f"成功复制 {copied_count} 个训练集文件对")
    
    # 复制验证集文件
    copied_count = 0
    for base_name in val_files:
        # 查找对应的图片文件
        image_ext = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if os.path.exists(os.path.join(images_dir, base_name + ext)):
                image_ext = ext
                break
        
        if image_ext is None:
            print(f"警告: 未找到 {base_name} 的图片文件")
            continue
        
        # 复制图片文件
        src_image = os.path.join(images_dir, base_name + image_ext)
        dst_image = os.path.join(output_dir, 'images', 'val', base_name + image_ext)
        shutil.copy2(src_image, dst_image)
        
        # 复制标签文件
        src_label = os.path.join(labels_dir, base_name + '.txt')
        dst_label = os.path.join(output_dir, 'labels', 'val', base_name + '.txt')
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
            copied_count += 1
        else:
            print(f"警告: 未找到标签文件 {src_label}")
    
    print(f"成功复制 {copied_count} 个验证集文件对")
    print("数据集准备完成！")
    
    # 打印一些统计信息
    train_images_count = len(os.listdir(os.path.join(output_dir, 'images', 'train')))
    val_images_count = len(os.listdir(os.path.join(output_dir, 'images', 'val')))
    train_labels_count = len(os.listdir(os.path.join(output_dir, 'labels', 'train')))
    val_labels_count = len(os.listdir(os.path.join(output_dir, 'labels', 'val')))
    
    print("\n最终统计:")
    print(f"训练集图片: {train_images_count} 张")
    print(f"训练集标签: {train_labels_count} 个")
    print(f"验证集图片: {val_images_count} 张")
    print(f"验证集标签: {val_labels_count} 个")

# 使用示例 - 请根据你的实际路径修改这些参数
if __name__ == "__main__":
    # 你的原始图片路径
    original_images_dir = "/home/rong/Pointnet_Pointnet2_pytorch/YOLO/tenniball and label/tennis"
    
    # 你的原始标签路径
    original_labels_dir = "/home/rong/Pointnet_Pointnet2_pytorch/YOLO/tenniball and label/label/labels_my-project-name_2025-09-05-10-35-05"
    
    # 输出数据集路径
    output_dataset_dir = "/home/rong/Pointnet_Pointnet2_pytorch/YOLO/tennis_dataset"
    
    # 执行数据集准备
    prepare_yolo_dataset(
        images_dir=original_images_dir,
        labels_dir=original_labels_dir,
        output_dir=output_dataset_dir,
        train_ratio=0.8,  # 80%训练，20%验证
        seed=42  # 随机种子，确保每次运行结果一致
    )