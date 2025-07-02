from train import compute_metrics, edit_distance

def test_tibetan_characters():
    """测试藏文字符的特殊情况"""
    print("=== 测试藏文字符特殊情况 ===")
    
    # 测试藏文复合字符
    test_cases = [
        # 基本藏文字符
        ("བ", "པ", 1),  # 不同的基字母
        ("བ", "བ", 0),  # 相同字符
        
        # 带元音符号的字符
        ("བི", "བུ", 1),  # 不同元音
        ("བི", "བི", 0),  # 相同复合字符
        
        # 复杂的藏文词
        ("བཀྲ་ཤིས", "བཀྲ་ཤིས", 0),  # 完全相同
        ("བཀྲ་ཤིས", "བཀྲ་ཤི", 1),   # 缺少一个字符
        ("བཀྲ་ཤིས", "བཀྲ་ཤིད", 1),   # 替换一个字符
        
        # 空格和标点
        ("བོད་", "བོད", 1),  # 有无标点
        ("བོད ཡིག", "བོདཡིག", 1),  # 有无空格
    ]
    
    print("藏文字符编辑距离测试:")
    for s1, s2, expected in test_cases:
        result = edit_distance(s1, s2)
        status = "✓" if result == expected else "✗"
        print(f"{status} edit_distance('{s1}', '{s2}') = {result}, 期望: {expected}")
        if result != expected:
            print(f"    字符串1长度: {len(s1)}, 字符串2长度: {len(s2)}")

def test_unicode_normalization():
    """测试Unicode规范化问题"""
    print("\n=== 测试Unicode规范化 ===")
    
    # 藏文字符可能有不同的Unicode表示方式
    # 这里测试一些可能的规范化问题
    
    # 测试字符串长度计算是否正确
    test_strings = [
        "བ",      # 单个基字母
        "བི",     # 基字母+元音
        "བཀྲ",    # 复合字符
        "བཀྲ་ཤིས", # 完整词汇
    ]
    
    print("字符串长度测试:")
    for s in test_strings:
        print(f"'{s}' -> 长度: {len(s)}")
        
    # 测试编辑距离是否受Unicode表示影响
    s1 = "བོད"
    s2 = "བོད"  # 看起来相同，但可能有不同的Unicode表示
    dist = edit_distance(s1, s2)
    print(f"\n相同藏文词的编辑距离: edit_distance('{s1}', '{s2}') = {dist}")
    if dist != 0:
        print("警告：相同的藏文词编辑距离不为0，可能存在Unicode规范化问题")

def test_edge_cases_tibetan():
    """测试藏文相关的边界情况"""
    print("\n=== 测试藏文边界情况 ===")
    
    # 测试非常长的藏文文本
    long_text1 = "བོད་ཡིག་གི་སྐད་ཡིག་ནི་བོད་མིའི་སྐད་ཡིག་ཡིན"
    long_text2 = "བོད་ཡིག་གི་སྐད་ཡིག་ནི་བོད་མིའི་སྐད་ཡིག་ཡིན"
    dist = edit_distance(long_text1, long_text2)
    print(f"长文本编辑距离: {dist} (期望: 0)")
    
    # 测试包含数字和英文的混合文本
    mixed1 = "བོད123abc"
    mixed2 = "བོད456def"
    dist = edit_distance(mixed1, mixed2)
    print(f"混合文本编辑距离: edit_distance('{mixed1}', '{mixed2}') = {dist}")
    
    # 测试评价指标对长文本的处理
    pred_texts = [long_text1, "བོད", ""]
    true_texts = [long_text2, "བོད", "ཡིག"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    print(f"\n长文本评价指标:")
    print(f"  WRR: {WRR:.2f}% ({correct}/{len(true_texts)})")
    print(f"  WRR_IF: {WRR_IF:.2f}% ({correct_one_error}/{len(true_texts)})")

if __name__ == '__main__':
    test_tibetan_characters()
    test_unicode_normalization()
    test_edge_cases_tibetan() 