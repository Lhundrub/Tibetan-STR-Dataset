import json
import os
from collections import Counter

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
    
    for entry in data:
        img_filename = os.path.basename(entry['image_path'])
        img_path = os.path.join('data/images', img_filename)
        
        if not os.path.exists(img_path):
            # 尝试简化文件名：去掉image后和-之前的部分
            if '-' in img_filename:
                simplified_filename = img_filename.split('-', 1)[1]  # 取-后面的部分
                simplified_path = os.path.join('data/images', simplified_filename)
                if os.path.exists(simplified_path):
                    img_path = simplified_path
                else:
                    print(f"简化文件名也不存在: {simplified_path}")
                    exit()
        
        transcription = entry['text']
        
        if isinstance(transcription, list):
            for text in transcription:
                if text and text.strip():
                    clean_text = text.strip().replace(' ', '')
                    all_texts.append(clean_text)
                    all_chars.update(clean_text)
        else:
            if transcription and transcription.strip():
                clean_text = transcription.strip().replace(' ', '')
                all_texts.append(clean_text)
                all_chars.update(clean_text)
    
    return all_chars, all_texts

def detailed_analysis():
    """详细的字符分析"""
    print("=== 详细字符分析报告 ===\n")
    
    # 加载词汇表
    vocab, char2idx = load_vocab('tibetan_vocab_full.txt')
    vocab_chars = set(vocab)
    
    # 获取数据集字符
    train_chars, train_texts = extract_chars_from_dataset('data/train_new.json')
    val_chars, val_texts = extract_chars_from_dataset('data/val_new.json')
    all_data_chars = train_chars | val_chars
    all_texts = train_texts + val_texts
    
    print(f"词汇表统计:")
    print(f"  总字符数: {len(vocab)}")
    
    # 分类词汇表字符
    special_tokens = [char for char in vocab if char.startswith('<')]
    tibetan_chars = [char for char in vocab if not char.startswith('<')]
    
    print(f"  特殊标记: {len(special_tokens)} 个")
    print(f"    {special_tokens}")
    print(f"  藏文字符: {len(tibetan_chars)} 个")
    
    print(f"\n数据集统计:")
    print(f"  训练集文本: {len(train_texts)} 个")
    print(f"  验证集文本: {len(val_texts)} 个")
    print(f"  总文本数: {len(all_texts)}")
    print(f"  数据中唯一字符: {len(all_data_chars)} 个")
    
    # 字符使用情况
    used_chars = all_data_chars & vocab_chars
    unused_chars = vocab_chars - all_data_chars
    
    print(f"\n字符使用情况:")
    print(f"  已使用的词汇表字符: {len(used_chars)} 个 ({len(used_chars)/len(vocab)*100:.1f}%)")
    print(f"  未使用的词汇表字符: {len(unused_chars)} 个 ({len(unused_chars)/len(vocab)*100:.1f}%)")
    
    # 详细显示未使用的字符
    print(f"\n未使用的词汇表字符详情:")
    unused_special = [char for char in unused_chars if char.startswith('<')]
    unused_tibetan = [char for char in unused_chars if not char.startswith('<')]
    
    print(f"  未使用的特殊标记: {len(unused_special)} 个")
    if unused_special:
        print(f"    {unused_special}")
    
    print(f"  未使用的藏文字符: {len(unused_tibetan)} 个")
    print(f"    {sorted(unused_tibetan)}")
    
    # 字符频率分析
    print(f"\n字符频率分析:")
    char_counter = Counter()
    for text in all_texts:
        char_counter.update(text)
    
    print(f"  最常用的20个字符:")
    for i, (char, count) in enumerate(char_counter.most_common(20), 1):
        print(f"    {i:2d}. '{char}': {count:4d} 次")
    
    # 低频字符分析
    low_freq_chars = [(char, count) for char, count in char_counter.items() if count <= 5]
    print(f"\n低频字符分析 (出现≤5次):")
    print(f"  低频字符数量: {len(low_freq_chars)}")
    if low_freq_chars:
        low_freq_chars.sort(key=lambda x: x[1])
        print(f"  低频字符列表:")
        for char, count in low_freq_chars:
            print(f"    '{char}': {count} 次")
    
    # 验证集独有字符分析
    val_only_chars = val_chars - train_chars
    if val_only_chars:
        print(f"\n验证集独有字符分析:")
        print(f"  验证集独有字符: {len(val_only_chars)} 个")
        val_only_freq = [(char, char_counter[char]) for char in val_only_chars]
        val_only_freq.sort(key=lambda x: x[1], reverse=True)
        for char, count in val_only_freq:
            print(f"    '{char}': {count} 次")
    
    # 优化建议
    print(f"\n=== 优化建议 ===")
    
    print(f"1. 词汇表优化:")
    if len(unused_tibetan) > 50:
        print(f"   - 可以考虑移除 {len(unused_tibetan)} 个未使用的藏文字符")
        print(f"   - 这将减少词汇表大小 {len(unused_tibetan)/len(vocab)*100:.1f}%")
    else:
        print(f"   - 未使用字符数量适中，建议保留以应对未来数据")
    
    print(f"2. 数据质量:")
    if val_only_chars:
        print(f"   - 验证集有 {len(val_only_chars)} 个训练集中没有的字符")
        print(f"   - 建议检查这些字符是否应该出现在训练集中")
    else:
        print(f"   - 验证集字符都在训练集中出现，数据一致性良好")
    
    print(f"3. 模型训练:")
    vocab_usage = len(used_chars) / len(vocab) * 100
    if vocab_usage < 50:
        print(f"   - 词汇表利用率较低 ({vocab_usage:.1f}%)，考虑精简词汇表")
    else:
        print(f"   - 词汇表利用率合理 ({vocab_usage:.1f}%)")
    
    # 保存优化后的词汇表建议
    print(f"\n=== 词汇表优化版本 ===")
    optimized_vocab = special_tokens + sorted(list(used_chars - set(special_tokens)))
    print(f"优化后词汇表大小: {len(optimized_vocab)} (原: {len(vocab)})")
    print(f"减少: {len(vocab) - len(optimized_vocab)} 个字符 ({(len(vocab) - len(optimized_vocab))/len(vocab)*100:.1f}%)")
    
    # 保存优化后的词汇表
    with open('tibetan_vocab_optimized.txt', 'w', encoding='utf-8') as f:
        for char in optimized_vocab:
            f.write(char + '\n')
    
    print(f"已保存优化词汇表到: tibetan_vocab_optimized.txt")

if __name__ == "__main__":
    detailed_analysis() 