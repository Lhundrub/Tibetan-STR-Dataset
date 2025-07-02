import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNNTransformer
from data_loader import OCRDataset, collate_fn
from train import compute_loss, decode_predictions
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

def quick_train_test():
    """快速训练测试，验证NaN问题已解决"""
    print("=== 快速训练测试 ===")
    
    # 使用优化词汇表
    vocab_path = 'tibetan_vocab_optimized.txt'
    
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"词汇表大小: {vocab_size}")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTransformer(vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 使用小批量数据进行测试
    train_dataset = OCRDataset('data/train.json', vocab_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    print(f"训练集样本数: {len(train_dataset)}")
    
    # 训练几个步骤
    model.train()
    
    valid_losses = []
    nan_count = 0
    total_batches = 0
    
    print("\n开始训练测试...")
    
    for batch_idx, (images, texts, text_lengths, raw_texts) in enumerate(train_loader):
        if batch_idx >= 10:  # 只测试10个batch
            break
            
        total_batches += 1
        
        images = images.to(device)
        texts = texts.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(images)
        
        # 计算损失
        loss = compute_loss(logits, texts, text_lengths, blank=0)
        
        # 检查损失
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            print(f"Batch {batch_idx + 1}: ❌ 损失无效 {loss.item()}")
        else:
            valid_losses.append(loss.item())
            print(f"Batch {batch_idx + 1}: ✅ 损失正常 {loss.item():.4f}")
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # 测试解码
            if batch_idx == 0:
                with torch.no_grad():
                    preds = decode_predictions(logits, idx2char, blank=0)
                    print(f"  真实文本: '{raw_texts[0]}'")
                    print(f"  预测文本: '{preds[0]}'")
    
    # 结果总结
    print(f"\n=== 测试结果 ===")
    print(f"总批次数: {total_batches}")
    print(f"有效损失数: {len(valid_losses)}")
    print(f"NaN错误数: {nan_count}")
    
    if nan_count == 0:
        print("🎉 所有损失都正常！NaN问题已解决。")
        if valid_losses:
            avg_loss = sum(valid_losses) / len(valid_losses)
            print(f"平均损失: {avg_loss:.4f}")
            print(f"损失范围: [{min(valid_losses):.4f}, {max(valid_losses):.4f}]")
        return True
    else:
        print(f"❌ 仍有 {nan_count} 个NaN错误")
        return False

if __name__ == "__main__":
    quick_train_test() 