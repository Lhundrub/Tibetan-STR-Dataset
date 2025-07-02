#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的藏文OCR代码
"""

import torch
import numpy as np
from model import CNNTransformer
from data_loader import OCRDataset, collate_fn
from train import compute_loss, decode_predictions
from torch.utils.data import DataLoader
from torchvision import transforms

def test_model_creation():
    """测试模型创建"""
    print("测试模型创建...")
    try:
        # 加载词汇表
        with open('tibetan_vocab_full.txt', 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        vocab_size = len(vocab)
        
        # 创建模型
        model = CNNTransformer(vocab_size=vocab_size)
        print(f"✓ 模型创建成功，词汇表大小: {vocab_size}")
        
        # 测试前向传播
        dummy_input = torch.randn(2, 1, 32, 128)  # [B, C, H, W]
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # 创建数据集（使用较小的数据集进行测试）
        dataset = OCRDataset('data/val.json', 'tibetan_vocab_full.txt', transform=transform)
        print(f"✓ 数据集创建成功，样本数量: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试单个样本
            sample = dataset[0]
            image, text_idx, text = sample
            print(f"✓ 样本加载成功，图像形状: {image.shape}, 文本: '{text}'")
            
            # 测试批处理
            loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
            for batch in loader:
                images, texts, text_lengths, raw_texts = batch
                print(f"✓ 批处理成功，批次图像形状: {images.shape}")
                break
        
        return True
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        return False

def test_loss_function():
    """测试损失函数"""
    print("\n测试CTC损失函数...")
    try:
        # 创建模拟数据
        T, B, V = 32, 2, 242  # 序列长度, 批次大小, 词汇表大小
        logits = torch.randn(T, B, V)
        
        # 模拟目标序列（不包含特殊标记）
        targets = torch.tensor([[4, 5, 6, 0, 0], [7, 8, 0, 0, 0]], dtype=torch.long)
        target_lengths = [3, 2]
        
        # 计算损失
        loss = compute_loss(logits, targets, target_lengths, blank=0)
        print(f"✓ CTC损失计算成功，损失值: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False

def test_decode_predictions():
    """测试预测解码"""
    print("\n测试预测解码...")
    try:
        # 加载词汇表
        with open('tibetan_vocab_full.txt', 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        idx2char = {idx: char for idx, char in enumerate(vocab)}
        
        # 创建模拟预测
        T, B, V = 10, 1, len(vocab)
        logits = torch.randn(T, B, V)
        
        # 解码预测
        decoded = decode_predictions(logits, idx2char, blank=0)
        print(f"✓ 预测解码成功，解码结果: {decoded}")
        
        return True
    except Exception as e:
        print(f"✗ 预测解码测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始测试修复后的藏文OCR代码...\n")
    
    tests = [
        test_model_creation,
        test_data_loader,
        test_loss_function,
        test_decode_predictions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("✓ 所有测试通过！代码修复成功。")
    else:
        print("✗ 部分测试失败，请检查相关问题。")

if __name__ == '__main__':
    main() 