import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import OCRDataset, collate_fn
import json
import argparse
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import os
import numpy as np
import glob
from datetime import datetime


# 计算CTC损失
def compute_loss(logits, targets, target_lengths, blank=0):
    """
    计算CTC损失
    logits: [T, B, V] - 模型输出的logits
    targets: [B, max_target_len] - 填充后的目标序列
    target_lengths: [B] - 每个序列的实际长度
    blank: blank标记的索引（在我们的词汇表中，<pad>=0作为blank）
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    input_lengths = torch.full(
        size=(logits.size(1),),
        fill_value=logits.size(0),
        dtype=torch.long,
        device=log_probs.device
    )
    
    # 将填充的目标序列转换为CTC期望的格式（连接的一维张量）
    targets_concat = []
    if not isinstance(target_lengths, torch.Tensor):
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=log_probs.device)
    
    for i, length in enumerate(target_lengths):
        # 只取实际长度的序列，不需要过滤特殊标记
        # 因为数据加载器已经不包含<sos>和<eos>了
        target_seq = targets[i][:length]
        targets_concat.extend(target_seq.tolist())
    
    targets_concat = torch.tensor(targets_concat, dtype=torch.long, device=log_probs.device)
    
    # 目标长度就是原始长度
    actual_target_lengths = target_lengths.clone()
    
    loss = nn.CTCLoss(blank=blank, zero_infinity=True)(
        log_probs, targets_concat, input_lengths, actual_target_lengths
    )
    return loss



# 解码预测结果
def decode_predictions(preds, idx2char, blank=0, remove_tokens=None):
    """
    CTC解码输出，去掉blank、重复字符，并移除特殊token（如<sos>、<eos>）。
    preds: [seq_len, batch_size, vocab_size] 或 [seq_len, batch_size]
    idx2char: 字典，索引到字符的映射
    blank: blank符号索引（通常为0）
    remove_tokens: 需要移除的特殊token字符列表（如['<sos>', '<eos>']）
    """
    if remove_tokens is None:
        remove_tokens = ['<sos>', '<eos>', '<pad>', '<unk>']
    decoded_texts = []
    # 如果是logits，先转为pred序列
    if preds.ndim == 3:
        preds = preds.argmax(dim=2)
    preds = preds.cpu().numpy()  # [seq_len, batch_size]
    
    for i in range(preds.shape[1]):
        raw_pred = preds[:, i]
        decoded = []
        prev_char_idx = None  # 初始化为None而不是blank
        
        for char_idx in raw_pred:
            # CTC解码规则：
            # 1. 跳过blank字符
            # 2. 跳过与前一个字符相同的字符（去重）
            if char_idx != blank and char_idx != prev_char_idx:
                char = idx2char.get(char_idx, '')
                if char not in remove_tokens:
                    decoded.append(char)
            prev_char_idx = char_idx
        
        decoded_texts.append(''.join(decoded))
    return decoded_texts



# 计算评估指标
def compute_metrics(pred_texts, true_texts):
    correct_words = 0
    correct_words_one_error = 0

    for pred, true in zip(pred_texts, true_texts):
        # 完全正确
        if pred == true:
            correct_words += 1
            correct_words_one_error += 1
            continue

        # 计算编辑距离
        dist = edit_distance(pred, true)

        # 允许一个字符错误（移除长度限制，因为空字符串也应该被考虑）
        if dist <= 1:
            correct_words_one_error += 1

    total_words = len(true_texts)
    if total_words == 0:
        return 0.0, 0.0, 0, 0
    
    WRR = (correct_words / total_words) * 100
    WRR_IF = (correct_words_one_error / total_words) * 100

    return WRR, WRR_IF, correct_words, correct_words_one_error


# 计算字符识别率 (Character Recognition Rate)
def compute_crr(pred_texts, true_texts):
    """
    计算字符级别的识别率
    CRR = (总字符数 - 编辑距离之和) / 总字符数 * 100%
    
    Args:
        pred_texts: 预测文本列表
        true_texts: 真实文本列表
    
    Returns:
        CRR: 字符识别率 (百分比)
        total_chars: 总字符数
        correct_chars: 正确字符数
    """
    if len(pred_texts) != len(true_texts):
        raise ValueError("预测文本和真实文本列表长度必须相同")
    
    total_chars = 0
    total_edit_distance = 0
    
    for pred, true in zip(pred_texts, true_texts):
        total_chars += len(true)  # 以真实文本的字符数为准
        total_edit_distance += edit_distance(pred, true)
    
    if total_chars == 0:
        return 0.0, 0, 0
    
    # 正确字符数 = 总字符数 - 编辑距离总和
    # 编辑距离表示需要多少次编辑操作来从预测转换为真实
    # 因此正确字符数可以近似为总字符数减去编辑距离
    correct_chars = max(0, total_chars - total_edit_distance)
    CRR = (correct_chars / total_chars) * 100
    
    return CRR, total_chars, correct_chars


# 编辑距离计算
def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def decode_label(label_tensor, idx2char):
    """
    输入: label_tensor - 单个标签序列 (Tensor 或 list[int])
    输出: 干净字符串（去掉<sos>、<eos>、<pad>）
    """
    if isinstance(label_tensor, torch.Tensor):
        idx_list = label_tensor.cpu().tolist()
    else:
        idx_list = list(label_tensor)
    # 去除<sos>、<eos>、<pad>
    tokens = []
    for idx in idx_list:
        char = idx2char.get(idx, '')
        if char in ('<sos>', '<eos>', '<pad>'):
            continue
        tokens.append(char)
    return ''.join(tokens)

