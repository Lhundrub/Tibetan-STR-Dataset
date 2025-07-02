import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from lightning.fabric import Fabric
from model_v0 import CNNTransformerBase

from data_loader import OCRDataset, collate_fn
from train import compute_loss, decode_predictions, compute_metrics, compute_crr
from torch.optim import Adam
import argparse
import os
import glob
from datetime import datetime
from ocr_transforms import get_ocr_transforms, get_light_augmentation, get_heavy_augmentation
import numpy as np

from line_profiler import profile
torch.set_float32_matmul_precision('high')

def compute_edit_distance_details(pred_text, true_text):
    """
    计算编辑距离及插入、删除、替换操作的数量
    
    Args:
        pred_text: 预测文本
        true_text: 真实文本
    
    Returns:
        edit_distance: 编辑距离
        insertions: 插入操作数量
        deletions: 删除操作数量
        substitutions: 替换操作数量
    """
    # 创建DP表格
    dp = np.zeros((len(true_text) + 1, len(pred_text) + 1), dtype=np.int32)
    
    # 操作类型表格: 0=无操作, 1=插入, 2=删除, 3=替换
    operations = np.zeros((len(true_text) + 1, len(pred_text) + 1), dtype=np.int32)
    
    # 初始化第一行和第一列
    for i in range(len(true_text) + 1):
        dp[i, 0] = i
        if i > 0:
            operations[i, 0] = 2  # 删除
    
    for j in range(len(pred_text) + 1):
        dp[0, j] = j
        if j > 0:
            operations[0, j] = 1  # 插入
    
    # 填充DP表格
    for i in range(1, len(true_text) + 1):
        for j in range(1, len(pred_text) + 1):
            if true_text[i-1] == pred_text[j-1]:
                dp[i, j] = dp[i-1, j-1]
                operations[i, j] = 0  # 匹配
            else:
                deletion = dp[i-1, j] + 1
                insertion = dp[i, j-1] + 1
                substitution = dp[i-1, j-1] + 1
                
                # 找到最小操作
                min_op = min(deletion, insertion, substitution)
                dp[i, j] = min_op
                
                if min_op == deletion:
                    operations[i, j] = 2  # 删除
                elif min_op == insertion:
                    operations[i, j] = 1  # 插入
                else:
                    operations[i, j] = 3  # 替换
    
    # 统计各操作数量
    i, j = len(true_text), len(pred_text)
    insertions, deletions, substitutions = 0, 0, 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and operations[i, j] == 0:  # 匹配
            i -= 1
            j -= 1
        elif j > 0 and operations[i, j] == 1:  # 插入
            insertions += 1
            j -= 1
        elif i > 0 and operations[i, j] == 2:  # 删除
            deletions += 1
            i -= 1
        elif i > 0 and j > 0 and operations[i, j] == 3:  # 替换
            substitutions += 1
            i -= 1
            j -= 1
        else:
            # 处理边界情况
            if i > 0:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1
    
    edit_distance = dp[len(true_text), len(pred_text)]
    return edit_distance, insertions, deletions, substitutions

def compute_error_rates(preds, targets):
    """
    计算插入错误率、删除错误率和替换错误率
    """
    total_chars = sum(len(t) for t in targets)
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    
    for pred, target in zip(preds, targets):
        _, ins, dels, subs = compute_edit_distance_details(pred, target)
        total_insertions += ins
        total_deletions += dels
        total_substitutions += subs
    
    insertion_error_rate = (total_insertions / total_chars) * 100 if total_chars > 0 else 0
    deletion_error_rate = (total_deletions / total_chars) * 100 if total_chars > 0 else 0
    substitution_error_rate = (total_substitutions / total_chars) * 100 if total_chars > 0 else 0
    
    return (insertion_error_rate, total_insertions, 
            deletion_error_rate, total_deletions, 
            substitution_error_rate, total_substitutions,
            total_chars)
@profile
def train_model(augmentation='heavy', epochs=100, batch_size=16, lr=0.0001):
    """
    使用Lightning Fabric训练模型 (16bit精度)
    """
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
    
    fabric.print(f"词汇表大小: {vocab_size}")
    fabric.print(f"数据增强模式: {augmentation}")
    fabric.print(f"混合精度训练: 16bit")
    
    # 初始化模型
    model = CNNTransformerBase(vocab_size)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 使用Fabric包装模型和优化器
    model, optimizer = fabric.setup(model, optimizer)
    
    # 加载预训练模型
    os.makedirs('saved_models', exist_ok=True)
    model_files = glob.glob("saved_models/best_model*.pth")
    
    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        fabric.print(f"🔍 发现预训练模型: {latest_model}")
        try:
            state_dict = fabric.load(latest_model)
            # model.load_state_dict(state_dict)
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
        'medium': get_ocr_transforms(train=True,img_height=128,img_width=512),
        'heavy': get_heavy_augmentation()
    }
    
    train_transform = transform_map.get(augmentation, get_ocr_transforms(train=True))
    val_transform = get_ocr_transforms(train=False)
    
    # 数据加载
    train_dataset = OCRDataset('data/train_new.json', vocab_path, transform=train_transform)
    val_dataset = OCRDataset('data/val_new.json', vocab_path, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2,persistent_workers=True)
    
    # 使用Fabric包装数据加载器
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)
    
    fabric.print(f"训练集样本数: {len(train_dataset)}")
    fabric.print(f"验证集样本数: {len(val_dataset)}")
    
    # 训练循环
    best_val_wrr = 0.0
    
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
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % 30 == 0:
                avg_loss = total_loss / num_batches
                fabric.print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}], Loss: {avg_loss:.4f}')
        
        # 每10个epoch完整验证
        if (epoch + 1) % 10 == 0:
            val_wrr = validate_model(fabric, model, val_loader, idx2char, epoch + 1)
            
            # 保存最佳模型
            if val_wrr > best_val_wrr:
                best_val_wrr = val_wrr
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f'saved_models/best_model_fabric_{timestamp}_{epoch}.pth'
                fabric.save(model_filename, model.state_dict())
                fabric.print(f'💾 保存最佳模型: {model_filename}, WRR: {val_wrr:.2f}%')
        else:
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            fabric.print(f'Epoch [{epoch + 1}/{epochs}] 训练损失: {avg_train_loss:.4f}')
    
    fabric.print("🎉 训练完成!")

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
    
    return WRR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lightning Fabric训练藏文OCR模型 (16bit)')
    parser.add_argument('--augmentation', '-a', 
                       choices=['none', 'light', 'medium', 'heavy'], 
                       default='medium',
                       help='数据增强强度 (default: heavy)')
    parser.add_argument('--epochs', '-e', type=int, default=300,
                       help='训练轮数 (default: 100)')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                       help='批次大小 (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率 (default: 0.0001)')
    
    args = parser.parse_args()
    
    print("=== Lightning Fabric 藏文OCR训练 ===")
    print(f"🚀 混合精度: 16bit")
    print(f"📈 数据增强: {args.augmentation}")
    print(f"🔄 训练轮数: {args.epochs}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📊 学习率: {args.lr}")
    print("=" * 35)
    
    train_model(
        augmentation=args.augmentation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    ) 