import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from lightning.fabric import Fabric
import argparse
import os
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 导入所有模型
from model2 import CNNTransformer
from model_improved1 import CNNTransformerResidual
from model_improved2 import CNNTransformerFPN
from model_improved3 import CNNTransformerAttention
from model_improved4 import CNNBiLSTMTransformer

# 导入训练所需工具
from data_loader import OCRDataset, collate_fn
from train import compute_loss, decode_predictions, compute_metrics, compute_crr
from train_fixed import compute_error_rates
from torch.optim import Adam
from ocr_transforms import get_ocr_transforms, get_light_augmentation, get_heavy_augmentation

# 设置高精度矩阵乘法
torch.set_float32_matmul_precision('high')

# 模型映射字典
MODEL_DICT = {
    'base': CNNTransformer,
    'residual': CNNTransformerResidual,
    'fpn': CNNTransformerFPN,
    'attention': CNNTransformerAttention,
    'bilstm': CNNBiLSTMTransformer
}

# 模型中文名称映射
MODEL_NAMES = {
    'base': '基础模型',
    'residual': '残差连接模型',
    'fpn': '特征金字塔模型',
    'attention': '注意力机制模型',
    'bilstm': '双向LSTM模型'
}

def train_model(model_name, augmentation='medium', epochs=100, batch_size=16, lr=0.0001):
    """
    使用Lightning Fabric训练指定模型
    
    Args:
        model_name: 要训练的模型名称
        augmentation: 数据增强类型
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
    
    Returns:
        字典，包含模型训练的各项指标
    """
    # 创建模型保存目录
    os.makedirs('saved_models', exist_ok=True)
    model_dir = Path(f'saved_models/{model_name}')
    model_dir.mkdir(exist_ok=True)
    
    # 初始化Fabric (启用16bit混合精度)
    fabric = Fabric(accelerator="auto", precision="16-mixed")
    fabric.launch()
    
    # 加载词汇表
    vocab_path = 'tibetan_vocab_optimized.txt'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    
    fabric.print(f"\n{'='*60}")
    fabric.print(f"开始训练: {MODEL_NAMES.get(model_name, model_name)}")
    fabric.print(f"词汇表大小: {vocab_size}")
    fabric.print(f"数据增强模式: {augmentation}")
    fabric.print(f"混合精度训练: 16bit")
    fabric.print(f"{'='*60}")
    
    # 初始化模型
    model_class = MODEL_DICT.get(model_name, CNNTransformer)
    model = model_class(vocab_size)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 使用Fabric包装模型和优化器
    model, optimizer = fabric.setup(model, optimizer)
    
    # 加载预训练模型（如果存在）
    model_files = glob.glob(str(model_dir / f"{model_name}_best_model*.pth"))
    
    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        fabric.print(f"🔍 发现预训练模型: {latest_model}")
        try:
            state_dict = fabric.load(latest_model)
            model.load_state_dict(state_dict)
            fabric.print("✅ 成功加载预训练模型，继续训练")
        except Exception as e:
            fabric.print(f"❌ 预训练模型加载失败: {e}")
            fabric.print("🆕 从头开始训练新模型")
    else:
        fabric.print("📝 未找到预训练模型文件")
        fabric.print("🆕 从头开始训练新模型")
    
    # 配置数据增强
    transform_map = {
        'none': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'light': get_light_augmentation(),
        'medium': get_ocr_transforms(train=True),
        'heavy': get_heavy_augmentation()
    }
    
    train_transform = transform_map.get(augmentation, get_ocr_transforms(train=True))
    val_transform = get_ocr_transforms(train=False)
    
    # 数据加载
    train_dataset = OCRDataset('data/train_new.json', vocab_path, transform=train_transform)
    val_dataset = OCRDataset('data/val_new.json', vocab_path, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, persistent_workers=True)
    
    # 使用Fabric包装数据加载器
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)
    
    fabric.print(f"训练集样本数: {len(train_dataset)}")
    fabric.print(f"验证集样本数: {len(val_dataset)}")
    
    # 训练循环
    best_val_wrr = 0.0
    results = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'wrr': [],
        'crr': [],
        'cer': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (images, texts, text_lengths, raw_texts) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(images)
            loss = compute_loss(logits, texts, text_lengths, blank=0)
            
            # 检查损失有效性
            if torch.isnan(loss) or torch.isinf(loss):
                fabric.print(f"⚠️ 损失无效: {loss.item()}")
                continue
            
            # 使用Fabric的反向传播 (自动处理16bit精度)
            fabric.backward(loss)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % 20 == 0:
                avg_loss = total_loss / num_batches
                fabric.print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}], Loss: {avg_loss:.4f}')
        
        # 记录训练损失
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        results['epoch'].append(epoch + 1)
        results['train_loss'].append(avg_train_loss)
        
        # 每5个epoch完整验证
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_metrics = validate_model(fabric, model, val_loader, idx2char, epoch + 1)
            
            # 记录验证指标
            results['val_loss'].append(val_metrics['val_loss'])
            results['wrr'].append(val_metrics['wrr'])
            results['crr'].append(val_metrics['crr'])
            results['cer'].append(val_metrics['cer'])
            
            # 保存最佳模型
            if val_metrics['wrr'] > best_val_wrr:
                best_val_wrr = val_metrics['wrr']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = str(model_dir / f"{model_name}_best_model_{timestamp}_{epoch}.pth")
                fabric.save(model_filename, model.state_dict())
                fabric.print(f'💾 保存最佳模型: {model_filename}, WRR: {val_metrics["wrr"]:.2f}%')
        else:
            fabric.print(f'Epoch [{epoch + 1}/{epochs}] 训练损失: {avg_train_loss:.4f}')
    
    fabric.print(f"🎉 {MODEL_NAMES.get(model_name, model_name)} 训练完成!")
    
    # 保存训练结果
    save_results(results, model_name)
    
    return results

def validate_model(fabric, model, val_loader, idx2char, epoch):
    """验证模型性能"""
    model.eval()
    all_preds = []
    all_true = []
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, texts, text_lengths, raw_texts in val_loader:
            logits = model(images)
            loss = compute_loss(logits, texts, text_lengths, blank=0)
            
            val_loss += loss.item()
            val_batches += 1
            
            preds = decode_predictions(logits, idx2char, blank=0)
            all_preds.extend(preds)
            all_true.extend(raw_texts)
    
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(all_preds, all_true)
    CRR, total_chars, correct_chars = compute_crr(all_preds, all_true)
    CER = 100.0 - CRR  # 字符错误率 = 100% - 字符正确率
    
    # 计算插入、删除和替换错误率
    (ins_err_rate, total_ins, 
     del_err_rate, total_del, 
     sub_err_rate, total_sub,
     _) = compute_error_rates(all_preds, all_true)
    
    fabric.print(f'\n📊 Epoch {epoch} 验证结果:')
    fabric.print(f'验证损失: {avg_val_loss:.4f}')
    fabric.print(f'WRR: {WRR:.2f}% ({correct}/{len(all_true)})')
    fabric.print(f'WRR_IF: {WRR_IF:.2f}% ({correct_one_error}/{len(all_true)})')
    fabric.print(f'CRR: {CRR:.2f}% ({correct_chars}/{total_chars})')
    fabric.print(f'CER: {CER:.2f}% ({total_chars-correct_chars}/{total_chars})')
    fabric.print(f'插入错误率: {ins_err_rate:.2f}% ({total_ins}/{total_chars})')
    fabric.print(f'删除错误率: {del_err_rate:.2f}% ({total_del}/{total_chars})')
    fabric.print(f'替换错误率: {sub_err_rate:.2f}% ({total_sub}/{total_chars})')
    
    # 显示预测示例
    fabric.print("📝 预测示例:")
    for i in range(min(3, len(all_preds))):
        fabric.print(f"  真实: '{all_true[i]}'")
        fabric.print(f"  预测: '{all_preds[i]}'")
    
    # 返回验证指标
    return {
        'val_loss': avg_val_loss,
        'wrr': WRR,
        'wrr_if': WRR_IF,
        'crr': CRR,
        'cer': CER,
        'ins_err': ins_err_rate,
        'del_err': del_err_rate,
        'sub_err': sub_err_rate
    }

def save_results(results, model_name):
    """保存训练结果到文件"""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # 保存训练数据
    np.save(results_dir / f"{model_name}_results.npy", results)
    
    # 生成并保存训练曲线
    plt.figure(figsize=(12, 10))
    
    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(results['epoch'], results['train_loss'], label='训练损失')
    if results['val_loss']:
        val_epochs = [results['epoch'][i] for i in range(len(results['epoch'])) if i % 5 == 0 or i == 0]
        plt.plot(val_epochs, results['val_loss'], 'o-', label='验证损失')
    plt.title(f'{MODEL_NAMES.get(model_name, model_name)} - 损失曲线')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 1, 2)
    if results['wrr']:
        val_epochs = [results['epoch'][i] for i in range(len(results['epoch'])) if i % 5 == 0 or i == 0]
        plt.plot(val_epochs, results['wrr'], 'o-', label='WRR')
        plt.plot(val_epochs, results['crr'], 'o-', label='CRR')
        plt.plot(val_epochs, results['cer'], 'o-', label='CER')
    plt.title(f'{MODEL_NAMES.get(model_name, model_name)} - 识别率曲线')
    plt.xlabel('轮数')
    plt.ylabel('百分比(%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_curves.png")

def compare_models(all_results):
    """比较不同模型的结果，生成比较图表"""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # 准备数据
    model_names = list(all_results.keys())
    final_wrr = [all_results[name]['wrr'][-1] for name in model_names]
    final_crr = [all_results[name]['crr'][-1] for name in model_names]
    final_cer = [all_results[name]['cer'][-1] for name in model_names]
    
    # 生成比较柱状图
    plt.figure(figsize=(14, 8))
    
    # WRR比较
    plt.subplot(1, 3, 1)
    x = np.arange(len(model_names))
    bar_width = 0.7
    plt.bar(x, final_wrr, width=bar_width, alpha=0.8, label='模型')
    plt.title('词识别率(WRR)比较')
    plt.xlabel('模型')
    plt.ylabel('WRR (%)')
    plt.xticks(x, [MODEL_NAMES.get(name, name) for name in model_names], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # CRR比较
    plt.subplot(1, 3, 2)
    plt.bar(x, final_crr, width=bar_width, alpha=0.8)
    plt.title('字符识别率(CRR)比较')
    plt.xlabel('模型')
    plt.ylabel('CRR (%)')
    plt.xticks(x, [MODEL_NAMES.get(name, name) for name in model_names], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # CER比较
    plt.subplot(1, 3, 3)
    plt.bar(x, final_cer, width=bar_width, alpha=0.8)
    plt.title('字符错误率(CER)比较')
    plt.xlabel('模型')
    plt.ylabel('CER (%)')
    plt.xticks(x, [MODEL_NAMES.get(name, name) for name in model_names], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison.png")
    
    # 打印比较表格
    print("\n\n模型性能比较表：")
    print("=" * 80)
    print(f"{'模型名称':<20} {'WRR (%)':<10} {'CRR (%)':<10} {'CER (%)':<10}")
    print("-" * 80)
    for i, name in enumerate(model_names):
        print(f"{MODEL_NAMES.get(name, name):<20} {final_wrr[i]:<10.2f} {final_crr[i]:<10.2f} {final_cer[i]:<10.2f}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='训练并比较多种藏文OCR模型')
    parser.add_argument('--models', '-m', nargs='+', 
                        choices=['base', 'residual', 'fpn', 'attention', 'bilstm', 'all'],
                        default=['all'],
                        help='要训练的模型 (default: all)')
    parser.add_argument('--augmentation', '-a', 
                       choices=['none', 'light', 'medium', 'heavy'], 
                       default='medium',
                       help='数据增强强度 (default: medium)')
    parser.add_argument('--epochs', '-e', type=int, default=300,
                       help='每个模型的训练轮数 (default: 50)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='批次大小 (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率 (default: 0.0001)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(f"{'藏文OCR多模型训练与比较':^60}")
    print("=" * 60)
    print(f"⚡ 运行环境: Windows")
    print(f"🚀 混合精度: 16bit")
    print(f"📈 数据增强: {args.augmentation}")
    print(f"🔄 每模型轮数: {args.epochs}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📊 学习率: {args.lr}")
    print("-" * 60)
    
    # 确定要训练的模型
    models_to_train = []
    if 'all' in args.models:
        models_to_train = list(MODEL_DICT.keys())
    else:
        models_to_train = args.models
    
    # 记录每个模型的结果
    all_results = {}
    total_start_time = time.time()
    
    # 训练每个模型
    for model_name in models_to_train:
        print(f"\n开始训练模型: {MODEL_NAMES.get(model_name, model_name)}")
        start_time = time.time()
        
        results = train_model(
            model_name=model_name,
            augmentation=args.augmentation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        all_results[model_name] = results
        
        elapsed = time.time() - start_time
        print(f"{MODEL_NAMES.get(model_name, model_name)} 训练完成，用时: {elapsed/60:.2f}分钟")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n所有模型训练完成，总用时: {total_elapsed/60:.2f}分钟")
    
    # 比较模型性能
    if len(models_to_train) > 1:
        compare_models(all_results)

if __name__ == '__main__':
    main() 