import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json

from model import CNNTransformer
from data_loader import OCRDataset, collate_fn

def debug_training():
    # 加载词汇表
    with open('tibetan_vocab_full.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"词汇表大小: {vocab_size}")
    print(f"前10个词汇: {vocab[:10]}")
    print(f"特殊标记索引: <pad>={char2idx.get('<pad>')}, <sos>={char2idx.get('<sos>')}, <eos>={char2idx.get('<eos>')}, <unk>={char2idx.get('<unk>')}")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = CNNTransformer(vocab_size).to(device)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_dataset = OCRDataset('data/val.json', 'tibetan_vocab_full.txt', transform=transform)
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 检查几个样本
    for i in range(min(3, len(val_dataset))):
        image, text_idx, raw_text = val_dataset[i]
        print(f"\n样本 {i}:")
        print(f"  原始文本: '{raw_text}'")
        print(f"  文本长度: {len(raw_text)}")
        print(f"  编码后: {text_idx.tolist()}")
        print(f"  编码长度: {len(text_idx)}")
        
        # 解码验证
        decoded_chars = [idx2char.get(idx.item(), '?') for idx in text_idx]
        decoded_text = ''.join(decoded_chars)
        print(f"  解码文本: '{decoded_text}'")
        print(f"  编码是否正确: {decoded_text == raw_text}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 测试一个batch
    model.eval()
    with torch.no_grad():
        for images, texts, text_lengths, raw_texts in val_loader:
            print(f"\n=== Batch 测试 ===")
            print(f"图像形状: {images.shape}")
            print(f"文本形状: {texts.shape}")
            print(f"文本长度: {text_lengths}")
            print(f"原始文本: {raw_texts}")
            
            images = images.to(device)
            logits = model(images)
            print(f"模型输出形状: {logits.shape}")  # 应该是 [T, B, V]
            
            # 检查输出分布
            probs = torch.softmax(logits, dim=2)
            max_probs, max_indices = torch.max(probs, dim=2)
            print(f"最大概率形状: {max_probs.shape}")
            print(f"最大索引形状: {max_indices.shape}")
            
            # 检查是否总是预测blank (索引0)
            blank_ratio = (max_indices == 0).float().mean()
            print(f"预测为blank的比例: {blank_ratio:.4f}")
            
            # 检查每个时间步的预测
            print(f"前5个时间步的预测索引:")
            for t in range(min(5, logits.shape[0])):
                print(f"  时间步 {t}: {max_indices[t].tolist()}")
            
            # 使用CTC解码
            from train import decode_predictions
            preds = decode_predictions(logits, idx2char, blank=0)
            print(f"CTC解码结果: {preds}")
            print(f"真实标签: {raw_texts}")
            
            # 检查是否有非空预测
            non_empty_preds = [p for p in preds if len(p) > 0]
            print(f"非空预测数量: {len(non_empty_preds)}/{len(preds)}")
            
            break  # 只测试第一个batch

if __name__ == '__main__':
    debug_training() 