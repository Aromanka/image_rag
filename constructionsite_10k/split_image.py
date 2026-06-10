import os
import shutil
import random
from pathlib import Path

def copy_images_equal_split(source_dir='images', target1='image1', target2='image2'):
    # 源目录
    src_path = Path(source_dir)
    if not src_path.exists() or not src_path.is_dir():
        print(f"错误：源目录 '{source_dir}' 不存在或不是文件夹")
        return
    
    # 支持的图片扩展名
    img_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    # 获取所有图片文件
    all_images = [f for f in src_path.iterdir() if f.is_file() and f.suffix.lower() in img_extensions]
    total = len(all_images)
    if total == 0:
        print("没有找到图片文件")
        return
    
    print(f"找到 {total} 张图片")
    
    # 创建目标文件夹
    target1_path = Path(target1)
    target2_path = Path(target2)
    target1_path.mkdir(exist_ok=True)
    target2_path.mkdir(exist_ok=True)
    
    # 随机打乱顺序，保证随机分配
    random.shuffle(all_images)
    
    # 计算每个文件夹应分配的数量
    half = total // 2
    # 如果是奇数，第一个文件夹多一张
    images1 = all_images[:half + (total % 2)]
    images2 = all_images[half + (total % 2):]
    
    print(f"image1 将复制 {len(images1)} 张图片")
    print(f"image2 将复制 {len(images2)} 张图片")
    
    # 复制函数
    def copy_files(file_list, dest_dir):
        for img in file_list:
            dest_path = dest_dir / img.name
            try:
                shutil.copy2(img, dest_path)  # copy2 保留元数据
            except Exception as e:
                print(f"复制 {img.name} 失败: {e}")
    
    print("正在复制到 image1...")
    copy_files(images1, target1_path)
    print("正在复制到 image2...")
    copy_files(images2, target2_path)
    
    print("完成！")

if __name__ == "__main__":
    copy_images_equal_split()