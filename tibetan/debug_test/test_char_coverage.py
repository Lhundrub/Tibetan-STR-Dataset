import json
import os
from collections import Counter, defaultdict

def load_vocab(vocab_path):
    """加载词汇表"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    return vocab, char2idx

def extract_chars_from_dataset(json_path):
    """从数据集中提取所有字符"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_chars = set()
    all_texts = []
    text_count = 0
    
    for entry in data:
        # 检查图像文件是否存在
        img_filename = os.path.basename(entry['ocr'])
        img_path = os.path.join('data/images', img_filename)
        
        if not os.path.exists(img_path):
            continue
        
        # 处理文本数据
        transcription = entry['transcription']
        
        if isinstance(transcription, list):
            # 如果是列表，处理每个文本
            for text in transcription:
                if text and text.strip():
                    clean_text = text.strip().replace(' ', '')
                    all_texts.append(clean_text)
                    all_chars.update(clean_text)
                    text_count += 1
        else:
            # 如果是字符串，直接处理
            if transcription and transcription.strip():
                clean_text = transcription.strip().replace(' ', '')
                all_texts.append(clean_text)
                all_chars.update(clean_text)
                text_count += 1
    
    return all_chars, all_texts, text_count

def analyze_char_coverage():
    """分析字符覆盖情况"""
    print("=== 藏文OCR字符覆盖分析 ===\n")
    
    # 加载词汇表
    vocab_path = 'tibetan_vocab_full.txt'
    vocab, char2idx = load_vocab(vocab_path)
    vocab_chars = set(vocab)
    
    print(f"词汇表大小: {len(vocab)}")
    print(f"特殊标记: {[v for v in vocab[:10] if v.startswith('<')]}")
    print()
    
    # 分析训练集
    print("--- 训练集分析 ---")
    try:
        train_chars, train_texts, train_count = extract_chars_from_dataset('data/train.json')
        print(f"训练集文本数量: {train_count}")
        print(f"训练集唯一字符数: {len(train_chars)}")
        
        # 找出训练集中不在词汇表中的字符
        train_missing_chars = train_chars - vocab_chars
        print(f"训练集中不在词汇表的字符数: {len(train_missing_chars)}")
        if train_missing_chars:
            print(f"缺失字符: {sorted(list(train_missing_chars))}")
        
        # 计算覆盖率
        train_coverage = (len(train_chars) - len(train_missing_chars)) / len(train_chars) * 100
        print(f"训练集字符覆盖率: {train_coverage:.2f}%")
        
    except Exception as e:
        print(f"训练集分析失败: {e}")
        train_chars = set()
        train_texts = []
    
    print()
    
    # 分析验证集
    print("--- 验证集分析 ---")
    try:
        val_chars, val_texts, val_count = extract_chars_from_dataset('data/val.json')
        print(f"验证集文本数量: {val_count}")
        print(f"验证集唯一字符数: {len(val_chars)}")
        
        # 找出验证集中不在词汇表中的字符
        val_missing_chars = val_chars - vocab_chars
        print(f"验证集中不在词汇表的字符数: {len(val_missing_chars)}")
        if val_missing_chars:
            print(f"缺失字符: {sorted(list(val_missing_chars))}")
        
        # 计算覆盖率
        val_coverage = (len(val_chars) - len(val_missing_chars)) / len(val_chars) * 100
        print(f"验证集字符覆盖率: {val_coverage:.2f}%")
        
        # 分析验证集中不在训练集中的字符
        if train_chars:
            val_not_in_train = val_chars - train_chars
            print(f"验证集中不在训练集的字符数: {len(val_not_in_train)}")
            if val_not_in_train:
                print(f"验证集独有字符: {sorted(list(val_not_in_train))}")
        
    except Exception as e:
        print(f"验证集分析失败: {e}")
        val_chars = set()
        val_texts = []
    
    print()
    
    # 综合分析
    print("--- 综合分析 ---")
    if train_chars and val_chars:
        all_data_chars = train_chars | val_chars
        all_missing_chars = all_data_chars - vocab_chars
        print(f"所有数据集唯一字符数: {len(all_data_chars)}")
        print(f"所有数据集中不在词汇表的字符数: {len(all_missing_chars)}")
        if all_missing_chars:
            print(f"所有缺失字符: {sorted(list(all_missing_chars))}")
        
        overall_coverage = (len(all_data_chars) - len(all_missing_chars)) / len(all_data_chars) * 100
        print(f"总体字符覆盖率: {overall_coverage:.2f}%")
        
        # 词汇表利用率
        used_vocab_chars = all_data_chars & vocab_chars
        vocab_usage = len(used_vocab_chars) / len(vocab_chars) * 100
        print(f"词汇表利用率: {vocab_usage:.2f}% ({len(used_vocab_chars)}/{len(vocab_chars)})")
        
        # 未使用的词汇表字符
        unused_vocab_chars = vocab_chars - all_data_chars
        special_tokens = {char for char in unused_vocab_chars if char.startswith('<')}
        unused_real_chars = unused_vocab_chars - special_tokens
        print(f"未使用的实际字符数: {len(unused_real_chars)}")
        if len(unused_real_chars) <= 20:  # 只显示少量未使用字符
            print(f"未使用的实际字符: {sorted(list(unused_real_chars))}")
    
    print()
    
    # 字符频率分析
    print("--- 字符频率分析 ---")
    if train_texts and val_texts:
        all_texts = train_texts + val_texts
        char_counter = Counter()
        for text in all_texts:
            char_counter.update(text)
        
        print(f"最常见的10个字符:")
        for char, count in char_counter.most_common(10):
            in_vocab = "✓" if char in vocab_chars else "✗"
            print(f"  '{char}': {count} 次 {in_vocab}")
        
        # 统计缺失字符的频率
        missing_chars = set(char_counter.keys()) - vocab_chars
        if missing_chars:
            print(f"\n缺失字符的使用频率:")
            missing_char_freq = [(char, char_counter[char]) for char in missing_chars]
            missing_char_freq.sort(key=lambda x: x[1], reverse=True)
            for char, count in missing_char_freq:
                print(f"  '{char}': {count} 次")

def analyze_text_length_distribution():
    """分析文本长度分布"""
    print("\n=== 文本长度分布分析 ===")
    
    datasets = [
        ('训练集', 'data/train.json'),
        ('验证集', 'data/val.json')
    ]
    
    for dataset_name, json_path in datasets:
        try:
            _, texts, _ = extract_chars_from_dataset(json_path)
            lengths = [len(text) for text in texts]
            
            if lengths:
                print(f"\n{dataset_name}:")
                print(f"  文本数量: {len(texts)}")
                print(f"  平均长度: {sum(lengths) / len(lengths):.2f}")
                print(f"  最短长度: {min(lengths)}")
                print(f"  最长长度: {max(lengths)}")
                print(f"  长度中位数: {sorted(lengths)[len(lengths)//2]}")
                
                # 长度分布
                length_dist = Counter(lengths)
                print(f"  最常见的5种长度:")
                for length, count in length_dist.most_common(5):
                    print(f"    长度{length}: {count}个文本")
                    
        except Exception as e:
            print(f"{dataset_name}分析失败: {e}")

if __name__ == "__main__":
    analyze_char_coverage()
    analyze_text_length_distribution() 