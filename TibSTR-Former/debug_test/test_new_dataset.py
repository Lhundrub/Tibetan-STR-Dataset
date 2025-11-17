import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import OCRDataset, collate_fn


def test_dataset_compatibility():
    print("=== 测试新数据集兼容性 ===")
    
    train_path = 'data/train_new.json'
    val_path = 'data/val_new.json'
    vocab_path = 'tibetan_vocab_full.txt'
    
    for path in [train_path, val_path, vocab_path]:
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return False
        else:
            print(f"✅ 文件存在: {path}")
    
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print("\n=== 测试数据集加载 ===")
        
        print("正在创建训练数据集...")
        train_dataset = OCRDataset(train_path, vocab_path, transform=transform)
        print(f"✅ 训练数据集创建成功: {len(train_dataset)} 样本")
        
        print("正在创建验证数据集...")
        val_dataset = OCRDataset(val_path, vocab_path, transform=transform)
        print(f"✅ 验证数据集创建成功: {len(val_dataset)} 样本")
        
        print("\n=== 测试DataLoader ===")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=4, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        print(f"✅ DataLoader创建成功")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        
        print("\n=== 测试数据批次 ===")
        
        for i, (images, texts, text_lengths, raw_texts) in enumerate(train_loader):
            print(f"批次 {i+1}:")
            print(f"  图像形状: {images.shape}")
            print(f"  文本形状: {texts.shape}")
            print(f"  文本长度: {text_lengths}")
            print(f"  原始文本数量: {len(raw_texts)}")
            print(f"  原始文本示例: {raw_texts[:2]}")
            
            assert isinstance(images, torch.Tensor), "图像应该是tensor类型"
            assert isinstance(texts, torch.Tensor), "文本应该是tensor类型"
            assert isinstance(text_lengths, list), "文本长度应该是列表类型"
            assert isinstance(raw_texts, list), "原始文本应该是列表类型"
            
            if i >= 2:
                break
        
        print("✅ 数据批次测试通过")
        
        print("\n=== 测试词汇表兼容性 ===")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        char2idx = {char: idx for idx, char in enumerate(vocab)}
        idx2char = {idx: char for idx, char in enumerate(vocab)}
        
        print(f"词汇表大小: {len(vocab)}")
        
        sample_texts = raw_texts[:3]
        encoded_successfully = 0
        
        for text in sample_texts:
            try:
                encoded = [char2idx.get(char, char2idx.get('<unk>', 0)) for char in text]
                decoded = ''.join([idx2char.get(idx, '') for idx in encoded])
                print(f"  原文: '{text}'")
                print(f"  编码: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
                print(f"  解码: '{decoded}'")
                encoded_successfully += 1
            except Exception as e:
                print(f"  ❌ 编码失败: {text} - {str(e)}")
        
        print(f"✅ 成功编码 {encoded_successfully}/{len(sample_texts)} 个样本")
        
        print("\n=== 性能测试 ===")
        
        import time
        start_time = time.time()
        batch_count = 0
        
        for batch in train_loader:
            batch_count += 1
            if batch_count >= 10:
                break
        
        elapsed_time = time.time() - start_time
        print(f"加载10个批次用时: {elapsed_time:.2f}秒")
        print(f"平均每批次: {elapsed_time/batch_count:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def compare_old_new_datasets():
    print("\n=== 比较旧数据集和新数据集 ===")
    
    try:
        old_train_path = 'data/train.json'
        old_val_path = 'data/val.json'
        new_train_path = 'data/train_new.json'
        new_val_path = 'data/val_new.json'
        
        datasets = {
            'old_train': old_train_path,
            'old_val': old_val_path,
            'new_train': new_train_path,
            'new_val': new_val_path
        }
        
        data_stats = {}
        
        for name, path in datasets.items():
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                total_samples = len(data)
                if data:
                    sample = data[0]
                    data_format = list(sample.keys())
                    
                    if 'text' in sample:
                        text_lengths = [len(item['text']) for item in data]
                    elif 'transcription' in sample:
                        text_lengths = []
                        for item in data:
                            if isinstance(item['transcription'], list):
                                text_lengths.extend([len(text) for text in item['transcription']])
                            else:
                                text_lengths.append(len(item['transcription']))
                    else:
                        text_lengths = []
                    
                    stats = {
                        'total_samples': total_samples,
                        'data_format': data_format,
                        'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                        'min_text_length': min(text_lengths) if text_lengths else 0,
                        'max_text_length': max(text_lengths) if text_lengths else 0,
                    }
                    
                    data_stats[name] = stats
                    print(f"\n{name}:")
                    print(f"  样本数: {stats['total_samples']}")
                    print(f"  数据格式: {stats['data_format']}")
                    print(f"  平均文本长度: {stats['avg_text_length']:.2f}")
                    print(f"  文本长度范围: {stats['min_text_length']} - {stats['max_text_length']}")
                else:
                    print(f"{name}: 空数据集")
            else:
                print(f"{name}: 文件不存在 ({path})")
        
        return data_stats
        
    except Exception as e:
        print(f"❌ 比较失败: {str(e)}")
        return None


def test_with_existing_training_code():
    print("\n=== 测试现有训练代码兼容性 ===")
    
    try:
        from model import CNNTransformer
        from train import compute_loss, decode_predictions
        
        vocab_path = 'tibetan_vocab_full.txt'
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        vocab_size = len(vocab)
        
        model = CNNTransformer(vocab_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print(f"✅ 模型创建成功，词汇表大小: {vocab_size}")
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        dataset = OCRDataset('data/train_new.json', vocab_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
        print("✅ 数据集和DataLoader创建成功")
        
        model.eval()
        char2idx = {char: idx for idx, char in enumerate(vocab)}
        idx2char = {idx: char for idx, char in enumerate(vocab)}
        
        for images, texts, text_lengths, raw_texts in dataloader:
            images = images.to(device)
            texts = texts.to(device)
            
            print(f"输入图像形状: {images.shape}")
            print(f"输入文本形状: {texts.shape}")
            
            with torch.no_grad():
                logits = model(images)
                print(f"模型输出形状: {logits.shape}")
                
                loss = compute_loss(logits, texts, text_lengths, blank=0)
                print(f"✅ 损失计算成功: {loss.item():.4f}")
                
                predictions = decode_predictions(logits, idx2char, blank=0)
                print(f"✅ 解码成功，预测结果: {predictions}")
                print(f"真实文本: {raw_texts}")
            
            break
        
        print("✅ 现有训练代码兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 兼容性测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=== 新数据集完整测试 ===")
    print("测试新划分的数据集是否能正确加载和使用")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    if test_dataset_compatibility():
        success_count += 1
        print("✅ 测试1通过: 数据集兼容性")
    else:
        print("❌ 测试1失败: 数据集兼容性")
    
    stats = compare_old_new_datasets()
    if stats:
        success_count += 1
        print("✅ 测试2通过: 数据集比较")
    else:
        print("❌ 测试2失败: 数据集比较")
    
    if test_with_existing_training_code():
        success_count += 1
        print("✅ 测试3通过: 训练代码兼容性")
    else:
        print("❌ 测试3失败: 训练代码兼容性")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！新数据集可以正常使用。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查问题。")
        return False


if __name__ == '__main__':
    main() 