import json
import os
from pathlib import Path


def debug_image_paths():
    """调试图片路径问题"""
    print("=== 调试图片路径问题 ===")
    
    # 检查新数据集的图片路径
    train_path = 'data/train_new.json'
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"训练集样本数: {len(train_data)}")
    
    # 检查前几个样本的图片路径
    sample_paths = []
    for i, sample in enumerate(train_data[:10]):
        img_path = sample['image_path']
        sample_paths.append(img_path)
        print(f"样本 {i+1}: {img_path}")
        
        # 检查各种可能的路径
        possible_paths = [
            img_path,  # 原始路径
            os.path.join('data', img_path),  # 加上data前缀
            img_path.replace('data/', ''),  # 去掉data前缀
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  ✅ 找到: {path}")
                found = True
                break
        
        if not found:
            print(f"  ❌ 未找到任何有效路径")
            
            # 检查目录结构
            path_parts = Path(img_path).parts
            print(f"  路径组成: {path_parts}")
            
            # 检查各级目录是否存在
            current_path = ""
            for part in path_parts:
                current_path = os.path.join(current_path, part)
                exists = os.path.exists(current_path)
                print(f"    {current_path}: {'存在' if exists else '不存在'}")
                if not exists:
                    break
    
    # 检查data目录结构
    print(f"\n=== data目录结构 ===")
    if os.path.exists('data'):
        for root, dirs, files in os.walk('data'):
            level = root.replace('data', '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for f in files[:5]:  # 只显示前5个文件
                print(f"{subindent}{f}")
            if len(files) > 5:
                print(f"{subindent}... 还有 {len(files) - 5} 个文件")
            if level > 3:  # 限制深度
                break
    else:
        print("data目录不存在!")
    
    return sample_paths


def check_image_directory_structure():
    """检查是否有其他可能的图片目录"""
    print(f"\n=== 检查其他可能的图片目录 ===")
    
    possible_dirs = [
        'images',
        'data/images', 
        'upload',
        'data/upload',
        'project-3-at-2025-04-12-19-03-81a58501',
        'data/project-3-at-2025-04-12-19-03-81a58501'
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            print(f"✅ 找到目录: {dir_path}")
            try:
                files = os.listdir(dir_path)
                print(f"  包含 {len(files)} 个文件/目录")
                if files:
                    print(f"  示例文件: {files[:3]}")
            except PermissionError:
                print(f"  无法访问该目录")
        else:
            print(f"❌ 目录不存在: {dir_path}")


def suggest_fixes(sample_paths):
    """建议修复方案"""
    print(f"\n=== 修复建议 ===")
    
    # 分析路径模式
    common_prefix = os.path.commonpath(sample_paths) if sample_paths else ""
    print(f"图片路径的公共前缀: {common_prefix}")
    
    # 检查是否需要创建符号链接或修改路径
    if sample_paths:
        first_path = sample_paths[0]
        path_parts = Path(first_path).parts
        
        print(f"\n建议的修复方案:")
        print(f"1. 如果图片在其他位置，创建符号链接:")
        print(f"   mklink /D \"data\\upload\" \"实际图片目录\"")
        
        print(f"2. 如果需要修改数据集路径，可以运行:")
        print(f"   python debug_test/fix_image_paths.py")
        
        print(f"3. 如果图片不存在，需要下载或获取图片文件")


if __name__ == '__main__':
    sample_paths = debug_image_paths()
    check_image_directory_structure()
    suggest_fixes(sample_paths) 