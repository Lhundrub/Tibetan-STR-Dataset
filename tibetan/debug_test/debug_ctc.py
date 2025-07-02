import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNNTransformer
from data_loader import OCRDataset, collate_fn
from train import compute_loss

def debug_ctc():
    # 加载词汇表
    with open('tibetan_vocab_full.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"词汇表大小: {vocab_size}")
    print(f"特殊标记: <pad>={char2idx['<pad>']}, <sos>={char2idx['<sos>']}, <eos>={char2idx['<eos>']}, <unk>={char2idx['<unk>']}")
    
    # 检查目标序列中是否包含blank标记
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_dataset = OCRDataset('data/val.json', 'tibetan_vocab_full.txt', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTransformer(vocab_size).to(device)
    
    # 测试一个batch
    for images, texts, text_lengths, raw_texts in val_loader:
        print(f"\n=== CTC调试 ===")
        print(f"原始文本: {raw_texts}")
        print(f"文本长度: {text_lengths}")
        print(f"编码后的文本形状: {texts.shape}")
        
        # 检查目标序列中的索引
        for i, (text, length) in enumerate(zip(texts, text_lengths)):
            actual_text = text[:length]
            print(f"\n样本 {i}: '{raw_texts[i]}'")
            print(f"  编码序列: {actual_text.tolist()}")
            print(f"  是否包含blank(0): {(actual_text == 0).any().item()}")
            print(f"  最小索引: {actual_text.min().item()}, 最大索引: {actual_text.max().item()}")
            
            # 解码验证
            decoded = ''.join([idx2char[idx.item()] for idx in actual_text])
            print(f"  解码结果: '{decoded}'")
            print(f"  解码正确: {decoded == raw_texts[i]}")
        
        # 测试模型前向传播
        images = images.to(device)
        texts = texts.to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(images)
            print(f"\n模型输出形状: {logits.shape}")
            
            # 计算损失
            try:
                loss = compute_loss(logits, texts, text_lengths, blank=0)
                print(f"CTC损失: {loss.item():.4f}")
                print("CTC损失计算成功！")
            except Exception as e:
                print(f"CTC损失计算失败: {e}")
                
                # 调试CTC输入
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
                
                # 构建目标序列
                targets_concat = []
                for i, length in enumerate(text_lengths):
                    target_seq = texts[i][:length]
                    targets_concat.extend(target_seq.tolist())
                
                targets_concat = torch.tensor(targets_concat, dtype=torch.long)
                actual_target_lengths = torch.tensor(text_lengths, dtype=torch.long)
                
                print(f"log_probs形状: {log_probs.shape}")
                print(f"input_lengths: {input_lengths}")
                print(f"targets_concat: {targets_concat}")
                print(f"actual_target_lengths: {actual_target_lengths}")
                print(f"targets_concat长度: {len(targets_concat)}")
                print(f"target_lengths总和: {sum(text_lengths)}")
        
        break  # 只测试第一个batch

if __name__ == '__main__':
    debug_ctc() 