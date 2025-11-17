import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNNTransformer
from data_loader import OCRDataset, collate_fn
from train import decode_predictions

def debug_decode():
    with open('tibetan_vocab_full.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"词汇表大小: {vocab_size}")
    print(f"blank标记: <pad>={char2idx['<pad>']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTransformer(vocab_size).to(device)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_dataset = OCRDataset('data/val.json', 'tibetan_vocab_full.txt', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    model.eval()
    with torch.no_grad():
        for images, texts, text_lengths, raw_texts in val_loader:
            print(f"\n=== 解码调试 ===")
            print(f"真实文本: {raw_texts}")
            
            images = images.to(device)
            logits = model(images)
            print(f"logits形状: {logits.shape}")
            
            probs = torch.softmax(logits, dim=2)
            max_probs, max_indices = torch.max(probs, dim=2)
            
            print(f"最大概率形状: {max_probs.shape}")
            print(f"最大索引形状: {max_indices.shape}")
            
            for b in range(logits.shape[1]):
                print(f"\n样本 {b}: '{raw_texts[b]}'")
                sample_indices = max_indices[:, b]
                sample_probs = max_probs[:, b]
                
                print(f"  预测索引序列: {sample_indices.tolist()}")
                print(f"  对应字符: {[idx2char.get(idx.item(), '?') for idx in sample_indices]}")
                print(f"  最大概率: {sample_probs.tolist()}")
                
                blank_count = (sample_indices == 0).sum().item()
                print(f"  blank数量: {blank_count}/{len(sample_indices)} ({blank_count/len(sample_indices)*100:.1f}%)")
                
                decoded_chars = []
                prev_idx = 0
                for idx in sample_indices:
                    idx_val = idx.item()
                    if idx_val != 0 and idx_val != prev_idx:
                        char = idx2char.get(idx_val, '?')
                        if char not in ['<pad>', '<sos>', '<eos>']:
                            decoded_chars.append(char)
                    prev_idx = idx_val
                
                manual_decoded = ''.join(decoded_chars)
                print(f"  手动解码: '{manual_decoded}'")
                
                original_decoded = decode_predictions(logits[:, b:b+1], idx2char, blank=0)
                print(f"  原始解码: {original_decoded}")
                
                non_blank_indices = sample_indices[sample_indices != 0]
                print(f"  非blank索引: {non_blank_indices.tolist()}")
                
                if len(non_blank_indices) == 0:
                    print(f"  问题: 所有预测都是blank!")
                elif len(set(non_blank_indices.tolist())) == 1:
                    print(f"  问题: 所有非blank预测都是同一个字符!")
            
            break

if __name__ == '__main__':
    debug_decode() 