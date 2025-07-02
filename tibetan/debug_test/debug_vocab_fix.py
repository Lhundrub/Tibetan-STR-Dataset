import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNNTransformer
from data_loader import OCRDataset, collate_fn
from train import compute_loss

def debug_vocab_fix():
    """验证词汇表修复"""
    print("=== 词汇表修复验证 ===")
    
    # 使用优化词汇表
    vocab_path = 'tibetan_vocab_optimized.txt'
    
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    
    print(f"📚 词汇表信息:")
    print(f"  文件: {vocab_path}")
    print(f"  大小: {vocab_size}")
    print(f"  前10个字符: {vocab[:10]}")
    print(f"  blank索引: {char2idx['<pad>']}")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTransformer(vocab_size).to(device)
    
    print(f"\n🤖 模型信息:")
    print(f"  设备: {device}")
    print(f"  词汇表大小: {vocab_size}")
    
    # 检查模型输出层
    if hasattr(model, 'classifier'):
        print(f"  模型输出层大小: {model.classifier.out_features}")
    
    # 数据加载器测试
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    print(f"\n📊 数据加载器测试:")
    try:
        # 使用小样本测试
        val_dataset = OCRDataset('data/val.json', vocab_path, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        print(f"  数据集大小: {len(val_dataset)}")
        print(f"  数据集词汇表大小: {len(val_dataset.char2idx)}")
        
        # 词汇表一致性检查
        if vocab_size == len(val_dataset.char2idx):
            print("  ✅ 词汇表大小一致")
        else:
            print(f"  ❌ 词汇表大小不一致: 模型({vocab_size}) vs 数据({len(val_dataset.char2idx)})")
            return False
        
        # 测试一个batch
        for batch_idx, (images, texts, text_lengths, raw_texts) in enumerate(val_loader):
            images = images.to(device)
            texts = texts.to(device)
            
            print(f"\n🔍 Batch {batch_idx + 1} 测试:")
            print(f"  图像形状: {images.shape}")
            print(f"  文本形状: {texts.shape}")
            print(f"  文本长度: {text_lengths}")
            print(f"  原始文本示例: {raw_texts[0]}")
            
            # 检查索引范围
            max_idx = texts.max().item()
            min_idx = texts.min().item()
            print(f"  文本索引范围: [{min_idx}, {max_idx}]")
            print(f"  词汇表范围: [0, {vocab_size-1}]")
            
            if max_idx >= vocab_size:
                print(f"  ❌ 索引超出范围: {max_idx} >= {vocab_size}")
                return False
            else:
                print("  ✅ 索引范围正常")
            
            # 测试模型前向传播
            model.eval()
            with torch.no_grad():
                try:
                    logits = model(images)
                    print(f"  模型输出形状: {logits.shape}")
                    print(f"  预期形状: [T, B, {vocab_size}]")
                    
                    if logits.shape[2] != vocab_size:
                        print(f"  ❌ 模型输出词汇表大小不匹配: {logits.shape[2]} != {vocab_size}")
                        return False
                    else:
                        print("  ✅ 模型输出形状正确")
                    
                    # 测试损失计算
                    loss = compute_loss(logits, texts, text_lengths, blank=0)
                    print(f"  损失值: {loss.item():.4f}")
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("  ❌ 损失计算异常")
                        return False
                    else:
                        print("  ✅ 损失计算正常")
                    
                    break
                    
                except Exception as e:
                    print(f"  ❌ 前向传播失败: {e}")
                    return False
            
            break
        
        print("\n🎉 所有测试通过！词汇表修复成功。")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    debug_vocab_fix() 