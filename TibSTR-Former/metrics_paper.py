"""
论文要求的4个评估指标实现
Paper Evaluation Metrics Implementation

根据论文附录A实现:
1. CER - Character Error Rate (字符错误率)
2. IER - Image Error Rate (图像错误率) 
3. DER - Diacritic Error Rate (变音符号错误率)
4. SER - Sentence Error Rate (句子错误率)
"""

import numpy as np
from typing import List, Tuple


TIBETAN_DIACRITICS = {
    'ི', 'ུ', 'ེ', 'ོ', 'ཀ', 'ཁ', 'ག', 'ང', 'ཅ', 'ཆ', 'ཇ', 'ཉ', 
    'ཏ', 'ཐ', 'ད', 'ན', 'པ', 'ཕ', 'བ', 'མ', 'ཙ', 'ཚ', 'ཛ', 'ཝ', 
    'ཞ', 'ཟ', 'འ', 'ཡ', 'ར', 'ལ', 'ཤ', 'ས', 'ཧ', 'ཨ', '༡', '༢', 
    '༣', '༤', '༥', '༦', '༧', '༨', '༩', '༠', '༼', '༽', '།', '༄༅།།', 
    '༈', '་', '༜', '༄', 'ྱ', 'ྲ', 'ྀ', 'ཽ', 'ཻ', 'ྭ', 'ཾ', 'ླ', 
    'ཥ', 'ཋ', 'ཌ', 'ཊ'
}


def compute_edit_distance_details(pred_text: str, true_text: str) -> Tuple[int, int, int, int]:
    """
    计算编辑距离及插入、删除、替换操作的数量
    
    Args:
        pred_text: 预测文本
        true_text: 真实文本
    
    Returns:
        edit_distance: 编辑距离
        insertions: 插入操作数量 (I)
        deletions: 删除操作数量 (D)
        substitutions: 替换操作数量 (S)
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


def compute_CER(pred_texts: List[str], true_texts: List[str]) -> Tuple[float, int, int, int, int]:
    """
    计算字符错误率 (Character Error Rate)
    
    公式: CER = (S + D + I) / N × 100%
    
    Args:
        pred_texts: 预测文本列表
        true_texts: 真实文本列表
    
    Returns:
        CER: 字符错误率 (%)
        S: 总替换次数
        D: 总删除次数
        I: 总插入次数
        N: 总字符数
    """
    if len(pred_texts) != len(true_texts):
        raise ValueError("预测文本和真实文本列表长度必须相同")
    
    total_chars = 0  # N
    total_substitutions = 0  # S
    total_deletions = 0  # D
    total_insertions = 0  # I
    
    for pred, true in zip(pred_texts, true_texts):
        total_chars += len(true)
        _, ins, dels, subs = compute_edit_distance_details(pred, true)
        total_substitutions += subs
        total_deletions += dels
        total_insertions += ins
    
    if total_chars == 0:
        return 0.0, 0, 0, 0, 0
    
    # CER = (S + D + I) / N
    CER = ((total_substitutions + total_deletions + total_insertions) / total_chars) * 100
    
    return CER, total_substitutions, total_deletions, total_insertions, total_chars


def compute_IER(pred_texts: List[str], true_texts: List[str]) -> Tuple[float, int, int]:
    """
    计算图像错误率 (Image Error Rate)
    
    公式: IER = |{i : ŷᵢ ≠ yᵢ}| / M × 100%
    
    Args:
        pred_texts: 预测文本列表
        true_texts: 真实文本列表
    
    Returns:
        IER: 图像错误率 (%)
        incorrect_images: 错误图像数量
        total_images: 总图像数量
    """
    if len(pred_texts) != len(true_texts):
        raise ValueError("预测文本和真实文本列表长度必须相同")
    
    total_images = len(true_texts)  # M
    incorrect_images = 0
    
    for pred, true in zip(pred_texts, true_texts):
        if pred != true:  # ŷᵢ ≠ yᵢ
            incorrect_images += 1
    
    if total_images == 0:
        return 0.0, 0, 0
    
    IER = (incorrect_images / total_images) * 100
    
    return IER, incorrect_images, total_images


def compute_DER(pred_texts: List[str], true_texts: List[str], 
                diacritics: set = None) -> Tuple[float, int, int]:
    """
    计算变音符号错误率 (Diacritic Error Rate)
    
    公式: DER = D_incorrect / D_total × 100%
    
    Args:
        pred_texts: 预测文本列表
        true_texts: 真实文本列表
        diacritics: 变音符号集合 (默认使用藏文变音符号)
    
    Returns:
        DER: 变音符号错误率 (%)
        incorrect_diacritics: 错误识别的变音符号数量
        total_diacritics: 真实标注中的变音符号总数
    """
    if diacritics is None:
        diacritics = TIBETAN_DIACRITICS
    
    if len(pred_texts) != len(true_texts):
        raise ValueError("预测文本和真实文本列表长度必须相同")
    
    total_diacritics = 0  # D_total
    incorrect_diacritics = 0  # D_incorrect
    
    for pred, true in zip(pred_texts, true_texts):
        # 提取真实文本中的变音符号位置
        true_diacritic_positions = []
        for i, char in enumerate(true):
            if char in diacritics:
                true_diacritic_positions.append((i, char))
                total_diacritics += 1
        
        # 提取预测文本中的变音符号
        pred_diacritics = []
        for i, char in enumerate(pred):
            if char in diacritics:
                pred_diacritics.append((i, char))
        
        # 对齐比较 (简化版本: 基于位置)
        # 更精确的方法需要使用编辑距离对齐
        pred_dict = {pos: char for pos, char in pred_diacritics}
        
        for pos, true_char in true_diacritic_positions:
            if pos not in pred_dict or pred_dict[pos] != true_char:
                incorrect_diacritics += 1
    
    if total_diacritics == 0:
        return 0.0, 0, 0
    
    DER = (incorrect_diacritics / total_diacritics) * 100
    
    return DER, incorrect_diacritics, total_diacritics


def compute_SER(pred_texts: List[str], true_texts: List[str]) -> Tuple[float, int, int]:
    """
    计算句子错误率 (Sentence Error Rate)
    
    公式: SER = |{k : ∃j, ŷₖ,ⱼ ≠ yₖ,ⱼ}| / K × 100%
    
    Args:
        pred_texts: 预测文本列表
        true_texts: 真实文本列表
    
    Returns:
        SER: 句子错误率 (%)
        incorrect_sentences: 错误句子数量
        total_sentences: 总句子数量
    """
    if len(pred_texts) != len(true_texts):
        raise ValueError("预测文本和真实文本列表长度必须相同")
    
    total_sentences = len(true_texts)  # K
    incorrect_sentences = 0
    
    for pred, true in zip(pred_texts, true_texts):
        # 如果存在任何字符不同，整个句子算错
        if pred != true:  # ∃j, ŷₖ,ⱼ ≠ yₖ,ⱼ
            incorrect_sentences += 1
    
    if total_sentences == 0:
        return 0.0, 0, 0
    
    SER = (incorrect_sentences / total_sentences) * 100
    
    return SER, incorrect_sentences, total_sentences


def compute_all_metrics(pred_texts: List[str], true_texts: List[str]) -> dict:
    """
    计算所有论文要求的评估指标
    
    Args:
        pred_texts: 预测文本列表
        true_texts: 真实文本列表
    
    Returns:
        metrics: 包含所有指标的字典
    """
    # 1. CER
    CER, S, D, I, N = compute_CER(pred_texts, true_texts)
    
    # 2. IER
    IER, incorrect_images, total_images = compute_IER(pred_texts, true_texts)
    
    # 3. DER
    DER, incorrect_diacritics, total_diacritics = compute_DER(pred_texts, true_texts)
    
    # 4. SER
    SER, incorrect_sentences, total_sentences = compute_SER(pred_texts, true_texts)
    
    metrics = {
        # 主要指标
        'CER': CER,
        'IER': IER,
        'DER': DER,
        'SER': SER,
        
        # CER详细信息
        'CER_substitutions': S,
        'CER_deletions': D,
        'CER_insertions': I,
        'CER_total_chars': N,
        
        # IER详细信息
        'IER_incorrect_images': incorrect_images,
        'IER_total_images': total_images,
        
        # DER详细信息
        'DER_incorrect_diacritics': incorrect_diacritics,
        'DER_total_diacritics': total_diacritics,
        
        # SER详细信息
        'SER_incorrect_sentences': incorrect_sentences,
        'SER_total_sentences': total_sentences,
        
        # 派生指标
        'accuracy_image': 100 - IER,  # 图像准确率
        'accuracy_sentence': 100 - SER,  # 句子准确率
        'accuracy_character': 100 - CER,  # 字符准确率
    }
    
    return metrics


def print_metrics(metrics: dict, detailed: bool = True):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        detailed: 是否打印详细信息
    """
    print("\n" + "="*60)
    print("论文评估指标 (Paper Evaluation Metrics)")
    print("="*60)
    
    # 主要指标
    print(f"\n📊 核心指标:")
    print(f"  CER (Character Error Rate):    {metrics['CER']:.2f}%")
    print(f"  IER (Image Error Rate):        {metrics['IER']:.2f}%")
    print(f"  DER (Diacritic Error Rate):    {metrics['DER']:.2f}%")
    print(f"  SER (Sentence Error Rate):     {metrics['SER']:.2f}%")
    
    if detailed:
        # CER详细信息
        print(f"\n📝 CER 详细:")
        print(f"  替换 (S): {metrics['CER_substitutions']}")
        print(f"  删除 (D): {metrics['CER_deletions']}")
        print(f"  插入 (I): {metrics['CER_insertions']}")
        print(f"  总字符 (N): {metrics['CER_total_chars']}")
        print(f"  公式: CER = (S+D+I)/N = ({metrics['CER_substitutions']}+{metrics['CER_deletions']}+{metrics['CER_insertions']})/{metrics['CER_total_chars']} = {metrics['CER']:.2f}%")
        
        # IER详细信息
        print(f"\n🖼️  IER 详细:")
        print(f"  错误图像: {metrics['IER_incorrect_images']}")
        print(f"  总图像: {metrics['IER_total_images']}")
        print(f"  正确图像: {metrics['IER_total_images'] - metrics['IER_incorrect_images']}")
        
        # DER详细信息
        print(f"\n🔤 DER 详细:")
        print(f"  错误变音符号: {metrics['DER_incorrect_diacritics']}")
        print(f"  总变音符号: {metrics['DER_total_diacritics']}")
        print(f"  正确变音符号: {metrics['DER_total_diacritics'] - metrics['DER_incorrect_diacritics']}")
        
        # SER详细信息
        print(f"\n📄 SER 详细:")
        print(f"  错误句子: {metrics['SER_incorrect_sentences']}")
        print(f"  总句子: {metrics['SER_total_sentences']}")
        print(f"  正确句子: {metrics['SER_total_sentences'] - metrics['SER_incorrect_sentences']}")
    
    # 准确率
    print(f"\n✅ 准确率:")
    print(f"  字符准确率: {metrics['accuracy_character']:.2f}%")
    print(f"  图像准确率: {metrics['accuracy_image']:.2f}%")
    print(f"  句子准确率: {metrics['accuracy_sentence']:.2f}%")
    
    print("="*60 + "\n")


# 使用示例
if __name__ == "__main__":
    # 测试数据
    true_texts = [
        "ཀ་ཁ་ག་ང་",
        "ཅིན་ཧྲི་བཟོ་རྩལ",
        "ཚ་ཚལ་རིགལ།",
        "ཀྲུང་ཧྭ་མི་རིགས"
    ]
    
    pred_texts = [
        "ཀ་ཁ་ག་ང་",      # 完全正确
        "ཅིན་ཧྲི་བཟོ་རྩ",  # 缺少最后一个字符
        "ཚ་ཚལ་རིག།",     # 缺少一个字符
        "ཀྲུང་ཧ་མི་རིགས"   # 一个字符错误
    ]
    
    print("测试数据:")
    for i, (true, pred) in enumerate(zip(true_texts, pred_texts)):
        match = "✅" if true == pred else "❌"
        print(f"  {i+1}. 真实: '{true}'")
        print(f"     预测: '{pred}' {match}")
    
    # 计算所有指标
    metrics = compute_all_metrics(pred_texts, true_texts)
    
    # 打印结果
    print_metrics(metrics, detailed=True)
    
    # 单独测试每个指标
    print("\n" + "="*60)
    print("单独指标测试")
    print("="*60)
    
    CER, S, D, I, N = compute_CER(pred_texts, true_texts)
    print(f"\nCER = {CER:.2f}% (S={S}, D={D}, I={I}, N={N})")
    
    IER, incorrect_imgs, total_imgs = compute_IER(pred_texts, true_texts)
    print(f"IER = {IER:.2f}% ({incorrect_imgs}/{total_imgs} 错误)")
    
    DER, incorrect_dia, total_dia = compute_DER(pred_texts, true_texts)
    print(f"DER = {DER:.2f}% ({incorrect_dia}/{total_dia} 错误)")
    
    SER, incorrect_sent, total_sent = compute_SER(pred_texts, true_texts)
    print(f"SER = {SER:.2f}% ({incorrect_sent}/{total_sent} 错误)")

