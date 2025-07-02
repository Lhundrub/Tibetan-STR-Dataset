from train import compute_metrics, edit_distance, compute_crr

def test_edit_distance():
    """测试编辑距离计算"""
    print("=== 测试编辑距离 ===")
    
    test_cases = [
        ("", "", 0),
        ("a", "", 1),
        ("", "a", 1),
        ("abc", "abc", 0),
        ("abc", "ab", 1),
        ("abc", "abcd", 1),
        ("abc", "adc", 1),
        ("abc", "def", 3),
        ("བོད", "བོད", 0),
        ("བོད", "བོ", 1),
        ("བོད", "བོས", 1),
        ("བོད", "དོད", 1),
    ]
    
    for s1, s2, expected in test_cases:
        result = edit_distance(s1, s2)
        status = "✓" if result == expected else "✗"
        print(f"{status} edit_distance('{s1}', '{s2}') = {result}, 期望: {expected}")

def test_compute_metrics():
    """测试评价指标计算"""
    print("\n=== 测试评价指标 ===")
    
    # 测试用例1：完全正确的情况
    pred_texts = ["བོད", "ཡར", "ཚོང"]
    true_texts = ["བོད", "ཡར", "ཚོང"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    CRR, total_chars, correct_chars = compute_crr(pred_texts, true_texts)
    print(f"测试1 - 完全正确:")
    print(f"  WRR: {WRR:.2f}% ({correct}/{len(true_texts)})")
    print(f"  WRR_IF: {WRR_IF:.2f}% ({correct_one_error}/{len(true_texts)})")
    print(f"  CRR: {CRR:.2f}% ({correct_chars}/{total_chars})")
    print(f"  期望: WRR=100%, WRR_IF=100%, CRR=100%")
    
    # 测试用例2：部分正确，部分一个字符错误
    pred_texts = ["བོད", "ཡ", "ཚོངས"]  # 第二个少一个字符，第三个多一个字符
    true_texts = ["བོད", "ཡར", "ཚོང"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    CRR, total_chars, correct_chars = compute_crr(pred_texts, true_texts)
    print(f"\n测试2 - 部分一个字符错误:")
    print(f"  WRR: {WRR:.2f}% ({correct}/{len(true_texts)})")
    print(f"  WRR_IF: {WRR_IF:.2f}% ({correct_one_error}/{len(true_texts)})")
    print(f"  CRR: {CRR:.2f}% ({correct_chars}/{total_chars})")
    print(f"  期望: WRR=33.33%, WRR_IF=100%, CRR=75%")
    
    # 测试用例3：完全错误
    pred_texts = ["ཀ", "ཁ", "ག"]
    true_texts = ["བོད", "ཡར", "ཚོང"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    CRR, total_chars, correct_chars = compute_crr(pred_texts, true_texts)
    print(f"\n测试3 - 完全错误:")
    print(f"  WRR: {WRR:.2f}% ({correct}/{len(true_texts)})")
    print(f"  WRR_IF: {WRR_IF:.2f}% ({correct_one_error}/{len(true_texts)})")
    print(f"  CRR: {CRR:.2f}% ({correct_chars}/{total_chars})")
    print(f"  期望: WRR=0%, WRR_IF=0%, CRR接近0%")
    
    # 测试用例4：空预测
    pred_texts = ["", "", ""]
    true_texts = ["བོད", "ཡར", "ཚོང"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    CRR, total_chars, correct_chars = compute_crr(pred_texts, true_texts)
    print(f"\n测试4 - 空预测:")
    print(f"  WRR: {WRR:.2f}% ({correct}/{len(true_texts)})")
    print(f"  WRR_IF: {WRR_IF:.2f}% ({correct_one_error}/{len(true_texts)})")
    print(f"  CRR: {CRR:.2f}% ({correct_chars}/{total_chars})")
    print(f"  期望: WRR=0%, WRR_IF=0%, CRR=0%")
    
    # 测试用例5：混合情况
    pred_texts = ["བོད", "ཡ", "ཚོངས", "ཀ", ""]
    true_texts = ["བོད", "ཡར", "ཚོང", "ཁ", "ག"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    CRR, total_chars, correct_chars = compute_crr(pred_texts, true_texts)
    print(f"\n测试5 - 混合情况:")
    print(f"  WRR: {WRR:.2f}% ({correct}/{len(true_texts)})")
    print(f"  WRR_IF: {WRR_IF:.2f}% ({correct_one_error}/{len(true_texts)})")
    print(f"  CRR: {CRR:.2f}% ({correct_chars}/{total_chars})")
    print(f"  期望: WRR=20%, WRR_IF=100%, CRR约42%")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 空列表
    WRR, WRR_IF, correct, correct_one_error = compute_metrics([], [])
    print(f"空列表: WRR={WRR}, WRR_IF={WRR_IF}")
    
    # 长度不匹配（这种情况不应该发生，但测试一下）
    try:
        pred_texts = ["བོད"]
        true_texts = ["བོད", "ཡར"]
        WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
        print(f"长度不匹配: 可能有问题")
    except Exception as e:
        print(f"长度不匹配处理: {e}")

def analyze_metric_issues():
    """分析可能的指标问题"""
    print("\n=== 分析可能的指标问题 ===")
    
    issues = []
    
    # 1. 检查编辑距离是否对称
    s1, s2 = "བོད", "བོས"
    dist1 = edit_distance(s1, s2)
    dist2 = edit_distance(s2, s1)
    if dist1 != dist2:
        issues.append(f"编辑距离不对称: edit_distance('{s1}', '{s2}')={dist1}, edit_distance('{s2}', '{s1}')={dist2}")
    
    # 2. 检查WRR_IF逻辑
    pred_texts = ["བོ"]  # 缺少一个字符
    true_texts = ["བོད"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    if WRR_IF != 100.0:
        issues.append(f"WRR_IF逻辑可能有问题: 一个字符差异应该被WRR_IF接受，但得到{WRR_IF}%")
    
    # 3. 检查空字符串处理
    pred_texts = [""]
    true_texts = ["བ"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    if WRR_IF != 100.0:  # 空字符串到一个字符的编辑距离是1
        issues.append(f"空字符串处理可能有问题: 空字符串到单字符应该被WRR_IF接受，但得到{WRR_IF}%")
    
    if issues:
        print("发现的问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("未发现明显问题")

if __name__ == '__main__':
    test_edit_distance()
    test_compute_metrics()
    test_edge_cases()
    analyze_metric_issues() 