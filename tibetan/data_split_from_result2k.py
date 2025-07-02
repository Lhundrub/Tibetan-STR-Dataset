import json
import os
import random
from collections import defaultdict
import argparse
from pathlib import Path


def analyze_result2k_data(json_path):
    """分析result2k.json的数据结构"""
    print(f"正在分析 {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 分析数据结构
    sample = data[0]
    print(f"第一个样本的键: {list(sample.keys())}")
    print(f"转录类型: {type(sample['transcription'])}")
    
    # 统计转录长度
    transcription_stats = defaultdict(int)
    total_text_regions = 0
    
    for item in data:
        if isinstance(item['transcription'], list):
            for text in item['transcription']:
                transcription_stats[len(text)] += 1
                total_text_regions += 1
        else:
            transcription_stats[len(item['transcription'])] += 1
            total_text_regions += 1
    
    print(f"总文本区域数: {total_text_regions}")
    print(f"转录长度分布 (前10):")
    for length, count in sorted(transcription_stats.items())[:10]:
        print(f"  长度 {length}: {count} 个")
    
    return data


def convert_to_ocr_format(data, output_dir):
    """将result2k格式转换为OCR训练格式"""
    print(f"正在转换数据格式...")
    
    ocr_samples = []
    
    for item in data:
        # 获取图片路径 - 去掉开头的路径分隔符
        image_path = item['ocr'].lstrip('/')
        
        # 处理转录文本
        if isinstance(item['transcription'], list):
            # 如果是列表，为每个文本区域创建一个样本
            for i, text in enumerate(item['transcription']):
                if i < len(item['bbox']):  # 确保有对应的bbox
                    bbox = item['bbox'][i]
                    ocr_sample = {
                        'image_path': image_path,
                        'text': text.strip(),
                        'bbox': bbox,
                        'id': f"{item['id']}_{i}",
                        'annotation_id': item['annotation_id']
                    }
                    ocr_samples.append(ocr_sample)
        else:
            # 如果是字符串，使用第一个bbox
            if item['bbox']:
                bbox = item['bbox'][0]
                ocr_sample = {
                    'image_path': image_path,
                    'text': item['transcription'].strip(),
                    'bbox': bbox,
                    'id': str(item['id']),
                    'annotation_id': item['annotation_id']
                }
                ocr_samples.append(ocr_sample)
    
    print(f"转换后的OCR样本数: {len(ocr_samples)}")
    return ocr_samples


def split_data(samples, train_ratio=0.8, val_ratio=0.2, random_seed=42):
    """划分训练集和验证集"""
    print(f"正在划分数据集 (训练集:{train_ratio}, 验证集:{val_ratio})...")
    
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 打乱数据
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    # 计算划分点
    total_samples = len(shuffled_samples)
    train_size = int(total_samples * train_ratio)
    
    # 划分数据
    train_samples = shuffled_samples[:train_size]
    val_samples = shuffled_samples[train_size:]
    
    print(f"训练集样本数: {len(train_samples)}")
    print(f"验证集样本数: {len(val_samples)}")
    
    return train_samples, val_samples


def save_splits(train_samples, val_samples, output_dir):
    """保存划分后的数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_path = os.path.join(output_dir, 'train_new.json')
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"训练集已保存到: {train_path}")
    
    # 保存验证集
    val_path = os.path.join(output_dir, 'val_new.json')
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"验证集已保存到: {val_path}")
    
    return train_path, val_path


def test_data_loading(train_path, val_path):
    """测试数据加载功能"""
    print("\n=== 测试数据加载 ===")
    
    try:
        # 测试加载训练集
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"✅ 成功加载训练集: {len(train_data)} 样本")
        
        # 测试加载验证集
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"✅ 成功加载验证集: {len(val_data)} 样本")
        
        # 检查数据格式
        if train_data:
            sample = train_data[0]
            required_keys = ['image_path', 'text', 'bbox', 'id']
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                print(f"❌ 训练集样本缺少必要字段: {missing_keys}")
                return False
            else:
                print("✅ 训练集数据格式正确")
        
        if val_data:
            sample = val_data[0]
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                print(f"❌ 验证集样本缺少必要字段: {missing_keys}")
                return False
            else:
                print("✅ 验证集数据格式正确")
        
        # 显示样本示例
        print(f"\n训练集示例:")
        print(f"  图片路径: {train_data[0]['image_path']}")
        print(f"  文本: '{train_data[0]['text']}'")
        print(f"  文本长度: {len(train_data[0]['text'])}")
        print(f"  边界框: {train_data[0]['bbox']}")
        
        print(f"\n验证集示例:")
        print(f"  图片路径: {val_data[0]['image_path']}")
        print(f"  文本: '{val_data[0]['text']}'")
        print(f"  文本长度: {len(val_data[0]['text'])}")
        print(f"  边界框: {val_data[0]['bbox']}")
        
        # 检查数据分布
        train_lengths = [len(sample['text']) for sample in train_data]
        val_lengths = [len(sample['text']) for sample in val_data]
        
        print(f"\n数据分布统计:")
        print(f"训练集文本长度: 最小={min(train_lengths)}, 最大={max(train_lengths)}, 平均={sum(train_lengths)/len(train_lengths):.2f}")
        print(f"验证集文本长度: 最小={min(val_lengths)}, 最大={max(val_lengths)}, 平均={sum(val_lengths)/len(val_lengths):.2f}")
        
        # 检查是否有重复
        train_ids = set(sample['id'] for sample in train_data)
        val_ids = set(sample['id'] for sample in val_data)
        overlap = train_ids.intersection(val_ids)
        
        if overlap:
            print(f"⚠️ 训练集和验证集有重复ID: {len(overlap)} 个")
        else:
            print("✅ 训练集和验证集无重复")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='从result2k.json重新划分训练和验证数据集')
    parser.add_argument('--input', '-i', default='data/result2k.json', 
                       help='输入的result2k.json文件路径')
    parser.add_argument('--output', '-o', default='data', 
                       help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.9, 
                       help='训练集比例 (默认: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1, 
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 检查比例是否合理
    if abs(args.train_ratio + args.val_ratio - 1.0) > 0.001:
        print(f"错误: 训练集和验证集比例之和必须为1.0，当前为{args.train_ratio + args.val_ratio}")
        return
    
    print("=== 从result2k.json重新划分数据集 ===")
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"随机种子: {args.seed}")
    print("=" * 50)
    
    # 1. 分析原始数据
    data = analyze_result2k_data(args.input)
    
    # 2. 转换数据格式
    ocr_samples = convert_to_ocr_format(data, args.output)
    
    # 3. 划分数据集
    train_samples, val_samples = split_data(
        ocr_samples, 
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        random_seed=args.seed
    )
    
    # 4. 保存数据集
    train_path, val_path = save_splits(train_samples, val_samples, args.output)
    
    # 5. 测试数据加载
    success = test_data_loading(train_path, val_path)
    
    if success:
        print("\n🎉 数据集划分和测试完成!")
        print(f"新的训练集: {train_path}")
        print(f"新的验证集: {val_path}")
    else:
        print("\n❌ 数据集划分过程中出现错误")


if __name__ == '__main__':
    main() 