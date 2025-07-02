import json
import os
import re
from collections import defaultdict


def get_suspicious_chars():
    """定义可疑的非藏文字符"""
    # 从词汇表补充中提取的可疑字符
    suspicious = [' ', '(', ')', '0', '3', '_', 'm', '༙', '༺', '྄', 'ཻ', 'ཽ']
    # 添加其他可能的非藏文字符
    suspicious.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    suspicious.extend(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    suspicious.extend(['1', '2', '4', '5', '6', '7', '8', '9'])
    return set(suspicious)


def is_tibetan_char(char):
    """判断字符是否为藏文字符"""
    # 藏文Unicode范围
    tibetan_ranges = [
        (0x0F00, 0x0FFF),  # 藏文基本区块
        (0x1000, 0x109F),  # 缅甸文区块（有些藏文扩展）
    ]
    
    char_code = ord(char)
    for start, end in tibetan_ranges:
        if start <= char_code <= end:
            return True
    
    # 允许的标点符号和空格
    allowed_chars = {' ', '།', '་', '༄', '༅', '༆', '༇', '༈', '༉', '༊', '༎', '༏', '༐', '༑', '༒', '༓', '༔'}
    return char in allowed_chars


def analyze_dataset_chars(json_path):
    """分析数据集中的字符"""
    print(f"正在分析 {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    suspicious_chars = get_suspicious_chars()
    char_occurrences = defaultdict(list)  # 字符 -> [(样本ID, 文本)]
    all_chars = set()
    
    for idx, entry in enumerate(data):
        # 获取文本内容
        if 'text' in entry:
            text = entry['text']
        elif 'transcription' in entry:
            if isinstance(entry['transcription'], list):
                text = ' '.join(entry['transcription'])
            else:
                text = entry['transcription']
        else:
            continue
        
        # 分析每个字符
        for char in text:
            all_chars.add(char)
            
            # 记录可疑字符
            if char in suspicious_chars or not is_tibetan_char(char):
                sample_id = entry.get('id', idx)
                char_occurrences[char].append((sample_id, text))
    
    return char_occurrences, all_chars


def print_suspicious_findings(char_occurrences, dataset_name):
    """打印可疑字符的发现"""
    print(f"\n=== {dataset_name} 中的可疑字符 ===")
    
    if not char_occurrences:
        print("✅ 未发现可疑字符")
        return
    
    # 按字符排序
    for char in sorted(char_occurrences.keys()):
        occurrences = char_occurrences[char]
        print(f"\n字符 '{char}' (Unicode: U+{ord(char):04X}) - 出现 {len(occurrences)} 次:")
        
        # 显示前几个样本
        for i, (sample_id, text) in enumerate(occurrences[:5]):
            print(f"  样本 {sample_id}: '{text}'")
        
        if len(occurrences) > 5:
            print(f"  ... 还有 {len(occurrences) - 5} 个样本")


def suggest_cleaning_actions(char_occurrences):
    """建议清理操作"""
    print(f"\n=== 数据清理建议 ===")
    
    if not char_occurrences:
        return
    
    # 分类可疑字符
    likely_errors = []
    possible_valid = []
    
    for char in char_occurrences:
        if char in ['(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'm', '_']:
            likely_errors.append(char)
        elif char in [' ']:  # 空格可能是合法的
            possible_valid.append(char)
        else:
            likely_errors.append(char)
    
    if likely_errors:
        print("需要清理的字符:")
        for char in likely_errors:
            count = len(char_occurrences[char])
            print(f"  '{char}' - {count} 次出现")
    
    if possible_valid:
        print("\n需要确认的字符:")
        for char in possible_valid:
            count = len(char_occurrences[char])
            print(f"  '{char}' - {count} 次出现")


def create_clean_dataset(json_path, output_path, char_occurrences):
    """创建清理后的数据集"""
    print(f"\n正在创建清理后的数据集...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = []
    removed_count = 0
    
    # 需要清理的字符
    chars_to_remove = {'(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'm', '_'}
    
    for entry in data:
        # 获取并清理文本
        if 'text' in entry:
            original_text = entry['text']
            # 移除可疑字符
            cleaned_text = ''.join(char for char in original_text if char not in chars_to_remove)
            # 清理多余空格
            cleaned_text = re.sub(r'\s+', '', cleaned_text)
            
            # 如果清理后文本为空或太短，跳过该样本
            if len(cleaned_text.strip()) < 2:
                removed_count += 1
                continue
            
            # 更新文本
            entry['text'] = cleaned_text
            cleaned_data.append(entry)
        elif 'transcription' in entry:
            # 处理旧格式
            if isinstance(entry['transcription'], list):
                cleaned_transcriptions = []
                for text in entry['transcription']:
                    cleaned_text = ''.join(char for char in text if char not in chars_to_remove)
                    cleaned_text = re.sub(r'\s+', '', cleaned_text)
                    if len(cleaned_text.strip()) >= 2:
                        cleaned_transcriptions.append(cleaned_text)
                
                if cleaned_transcriptions:
                    entry['transcription'] = cleaned_transcriptions
                    cleaned_data.append(entry)
                else:
                    removed_count += 1
            else:
                cleaned_text = ''.join(char for char in entry['transcription'] if char not in chars_to_remove)
                cleaned_text = re.sub(r'\s+', '', cleaned_text)
                if len(cleaned_text.strip()) >= 2:
                    entry['transcription'] = cleaned_text
                    cleaned_data.append(entry)
                else:
                    removed_count += 1
        else:
            cleaned_data.append(entry)
    
    # 保存清理后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"清理完成:")
    print(f"  原始样本: {len(data)}")
    print(f"  清理后样本: {len(cleaned_data)}")
    print(f"  移除样本: {removed_count}")
    print(f"  保存到: {output_path}")


def main():
    """主函数"""
    print("=== 检查数据集中的无效字符 ===")
    
    datasets = [
        ('data/train_new.json', '训练集'),
        ('data/val_new.json', '验证集')
    ]
    
    all_suspicious = {}
    
    for json_path, name in datasets:
        if os.path.exists(json_path):
            char_occurrences, all_chars = analyze_dataset_chars(json_path)
            print_suspicious_findings(char_occurrences, name)
            all_suspicious[name] = char_occurrences
        else:
            print(f"文件不存在: {json_path}")
    
    # 综合建议
    if any(all_suspicious.values()):
        all_chars = {}
        for dataset_chars in all_suspicious.values():
            for char, occurrences in dataset_chars.items():
                if char not in all_chars:
                    all_chars[char] = []
                all_chars[char].extend(occurrences)
        
        suggest_cleaning_actions(all_chars)
        
        # 询问是否创建清理后的数据集
        print(f"\n是否创建清理后的数据集？")
        print(f"这将移除可疑字符并保存为 train_cleaned.json 和 val_cleaned.json")
        
        # 自动创建清理版本用于测试
        for json_path, name in datasets:
            if os.path.exists(json_path):
                output_path = json_path.replace('.json', '_cleaned.json')
                char_occurrences = all_suspicious[name]
                create_clean_dataset(json_path, output_path, char_occurrences)


if __name__ == '__main__':
    main() 