from train import compute_metrics, edit_distance

def test_tibetan_characters():
    print("=== 测试藏文字符特殊情况 ===")
    
    test_cases = [
        ("བ", "པ", 1),
        ("བ", "བ", 0),
        ("བི", "བུ", 1),
        ("བི", "བི", 0),
        ("བཀྲ་ཤིས", "བཀྲ་ཤིས", 0),
        ("བཀྲ་ཤིས", "བཀྲ་ཤི", 1),
        ("བཀྲ་ཤིས", "བཀྲ་ཤིད", 1),
        ("བོད་", "བོད", 1),
        ("བོད ཡིག", "བོདཡིག", 1),
    ]
    
    print("藏文字符编辑距离测试:")
    for s1, s2, expected in test_cases:
        result = edit_distance(s1, s2)
        status = "✓" if result == expected else "✗"
        print(f"{status} edit_distance('{s1}', '{s2}') = {result}, 期望: {expected}")
        if result != expected:
            print(f"    字符串1长度: {len(s1)}, 字符串2长度: {len(s2)}")

def test_unicode_normalization():
    print("\n=== 测试Unicode规范化 ===")
    
    test_strings = [
        "བ",
        "བི",
        "བཀྲ",
        "བཀྲ་ཤིས",
    ]
    
    print("字符串长度测试:")
    for s in test_strings:
        print(f"'{s}' -> 长度: {len(s)}")
        
    s1 = "བོད"
    s2 = "བོད"
    dist = edit_distance(s1, s2)
    print(f"\n相同藏文词的编辑距离: edit_distance('{s1}', '{s2}') = {dist}")
    if dist != 0:
        print("警告：相同的藏文词编辑距离不为0，可能存在Unicode规范化问题")

def test_edge_cases_tibetan():
    print("\n=== 测试藏文边界情况 ===")
    
    long_text1 = "བོད་ཡིག་གི་སྐད་ཡིག་ནི་བོད་མིའི་སྐད་ཡིག་ཡིན"
    long_text2 = "བོད་ཡིག་གི་སྐད་ཡིག་ནི་བོད་མིའི་སྐད་ཡིག་ཡིན"
    dist = edit_distance(long_text1, long_text2)
    print(f"长文本编辑距离: {dist} (期望: 0)")
    
    mixed1 = "བོད123abc"
    mixed2 = "བོད456def"
    dist = edit_distance(mixed1, mixed2)
    print(f"混合文本编辑距离: edit_distance('{mixed1}', '{mixed2}') = {dist}")
    
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