import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loader import OCRDataset
from ocr_transforms import get_ocr_transforms, get_light_augmentation, get_heavy_augmentation
import torchvision.transforms as transforms

def test_augmentation_effects():
    """测试和可视化数据增强效果"""
    print("=== 数据增强效果测试 ===")
    
    # 不同的增强配置
    augmentations = {
        'none': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'light': get_light_augmentation(),
        'medium': get_ocr_transforms(train=True),
        'heavy': get_heavy_augmentation()
    }
    
    vocab_path = 'tibetan_vocab_optimized.txt'
    
    # 为每种增强创建数据集
    datasets = {}
    for name, transform in augmentations.items():
        try:
            dataset = OCRDataset('data/val.json', vocab_path, transform=transform)
            datasets[name] = dataset
            print(f"✅ {name} 增强数据集创建成功，样本数: {len(dataset)}")
        except Exception as e:
            print(f"❌ {name} 增强数据集创建失败: {e}")
    
    if not datasets:
        print("❌ 没有可用的数据集")
        return
    
    # 选择同一个样本进行对比
    sample_idx = 0
    
    print(f"\n🔍 对比样本 {sample_idx}:")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('数据增强效果对比', fontsize=16)
    
    for i, (name, dataset) in enumerate(datasets.items()):
        if i >= 8:  # 最多显示8个
            break
            
        try:
            # 获取样本
            img_tensor, text_idx, raw_text = dataset[sample_idx]
            
            # 转换tensor为numpy用于显示
            if isinstance(img_tensor, torch.Tensor):
                img = img_tensor.squeeze().numpy()
                # 反标准化
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
            else:
                img = img_tensor
            
            # 显示图像
            row = i // 4
            col = i % 4
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'{name.capitalize()}')
            axes[row, col].axis('off')
            
            print(f"  {name}: 文本='{raw_text}', 形状={img.shape}")
            
        except Exception as e:
            print(f"  ❌ {name} 处理失败: {e}")
            axes[i//4, i%4].text(0.5, 0.5, f'Error\n{name}', 
                                ha='center', va='center', transform=axes[i//4, i%4].transAxes)
            axes[i//4, i%4].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(datasets), 8):
        axes[i//4, i%4].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📸 增强对比图已保存: augmentation_comparison.png")

def test_multiple_samples():
    """测试同一种增强的多个样本"""
    print("\n=== 中等强度增强多样本测试 ===")
    
    vocab_path = 'tibetan_vocab_optimized.txt'
    transform = get_ocr_transforms(train=True)
    
    try:
        dataset = OCRDataset('data/val.json', vocab_path, transform=transform)
        
        # 选择前6个样本
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('中等强度数据增强样本展示', fontsize=16)
        
        for i in range(6):
            if i >= len(dataset):
                break
                
            img_tensor, text_idx, raw_text = dataset[i]
            
            # 转换显示
            if isinstance(img_tensor, torch.Tensor):
                img = img_tensor.squeeze().numpy()
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
            else:
                img = img_tensor
            
            row = i // 3
            col = i % 3
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f"'{raw_text}'", fontsize=10)
            axes[row, col].axis('off')
            
            print(f"  样本 {i+1}: '{raw_text}'")
        
        plt.tight_layout()
        plt.savefig('medium_augmentation_samples.png', dpi=150, bbox_inches='tight')
        print(f"📸 多样本展示已保存: medium_augmentation_samples.png")
        
    except Exception as e:
        print(f"❌ 多样本测试失败: {e}")

def analyze_augmentation_impact():
    """分析数据增强对训练的潜在影响"""
    print("\n=== 数据增强影响分析 ===")
    
    vocab_path = 'tibetan_vocab_optimized.txt'
    
    # 测试不同增强的batch
    augmentations = {
        'none': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'medium': get_ocr_transforms(train=True)
    }
    
    from torch.utils.data import DataLoader
    from data_loader import collate_fn
    
    for name, transform in augmentations.items():
        print(f"\n📊 {name.capitalize()} 增强分析:")
        
        try:
            dataset = OCRDataset('data/val.json', vocab_path, transform=transform)
            loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
            
            # 获取一个batch
            for images, texts, text_lengths, raw_texts in loader:
                print(f"  Batch形状: {images.shape}")
                print(f"  图像值范围: [{images.min().item():.3f}, {images.max().item():.3f}]")
                print(f"  图像均值: {images.mean().item():.3f}")
                print(f"  图像标准差: {images.std().item():.3f}")
                print(f"  文本示例: {raw_texts[0]}")
                break
                
        except Exception as e:
            print(f"  ❌ 分析失败: {e}")
    
    print("\n💡 建议:")
    print("  - none: 适用于高质量、一致的数据")
    print("  - light: 适用于数据量较大的情况")
    print("  - medium: 平衡的选择，适用于大多数情况")
    print("  - heavy: 适用于数据量小或质量不一的情况")

if __name__ == "__main__":
    test_augmentation_effects()
    test_multiple_samples()
    analyze_augmentation_impact() 