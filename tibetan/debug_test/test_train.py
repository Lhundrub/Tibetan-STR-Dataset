import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNNTransformer
from data_loader import OCRDataset, collate_fn
from train import compute_loss, decode_predictions, compute_metrics
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

def test_training():
    # 加载词汇表
    with open('tibetan_vocab_full.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"词汇表大小: {vocab_size}")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = CNNTransformer(vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 使用小批量数据进行快速测试
    val_dataset = OCRDataset('data/val.json', 'tibetan_vocab_full.txt', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 训练几个步骤
    print("\n开始训练测试...")
    model.train()
    
    for epoch in range(30):  # 只训练3个epoch
        total_loss = 0
        num_batches = 0
        
        for i, (images, texts, text_lengths, raw_texts) in enumerate(val_loader):
            # if i >= 5:  # 只训练前5个batch
            #     break
                
            images = images.to(device)
            texts = texts.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(images)
            
            # 计算损失
            loss = compute_loss(logits, texts, text_lengths, blank=0)
            
            # 反向传播
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % 2 == 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.4f}')
        
        # 验证
        print(f"\nEpoch {epoch + 1} 验证:")
        model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for i, (images, texts, text_lengths, raw_texts) in enumerate(val_loader):
                # if i >= 3:  # 只验证前3个batch
                    # break
                    
                images = images.to(device)
                logits = model(images)
                
                preds = decode_predictions(logits, idx2char, blank=0)
                all_preds.extend(preds)
                all_true.extend(raw_texts)
        
        # 计算指标
        from train import compute_metrics
        WRR, WRR_IF, correct, correct_one_error = compute_metrics(all_preds, all_true)
        print(f'WRR: {WRR:.2f}% ({correct}/{len(all_true)})')
        print(f'WRR_IF: {WRR_IF:.2f}% ({correct_one_error}/{len(all_true)})')
        
        # 显示一些预测示例
        print("预测示例:")
        for i in range(min(3, len(all_preds))):
            print(f"  真实: '{all_true[i]}'")
            print(f"  预测: '{all_preds[i]}'")
            print()
        
        model.train()
    
    print("训练测试完成！")

if __name__ == '__main__':
    test_training() 