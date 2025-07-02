# 简化的编辑距离计算
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

# 修复后的评价指标
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

        # 允许一个字符错误（移除长度限制）
        if dist <= 1:
            correct_words_one_error += 1

    total_words = len(true_texts)
    if total_words == 0:
        return 0.0, 0.0, 0, 0
    
    WRR = (correct_words / total_words) * 100
    WRR_IF = (correct_words_one_error / total_words) * 100

    return WRR, WRR_IF, correct_words, correct_words_one_error

def test_metrics():
    print("=== 测试修复后的评价指标 ===")
    
    # 测试用例1：完全正确
    pred_texts = ["བོད", "ཡར", "ཚོང"]
    true_texts = ["བོད", "ཡར", "ཚོང"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    print(f"测试1 - 完全正确: WRR={WRR:.1f}%, WRR_IF={WRR_IF:.1f}% (期望: 100%, 100%)")
    
    # 测试用例2：一个字符错误
    pred_texts = ["བོད", "ཡ", "ཚོངས"]  # 第二个少一个字符，第三个多一个字符
    true_texts = ["བོད", "ཡར", "ཚོང"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    print(f"测试2 - 一个字符错误: WRR={WRR:.1f}%, WRR_IF={WRR_IF:.1f}% (期望: 33.3%, 100%)")
    
    # 测试用例3：空预测
    pred_texts = ["", "ཡ", ""]
    true_texts = ["བ", "ཡར", "ཚ"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    print(f"测试3 - 包含空预测: WRR={WRR:.1f}%, WRR_IF={WRR_IF:.1f}% (期望: 0%, 100%)")
    
    # 测试用例4：完全错误
    pred_texts = ["ཀཀཀ", "ཁཁཁ", "གགགག"]
    true_texts = ["བོད", "ཡར", "ཚོང"]
    WRR, WRR_IF, correct, correct_one_error = compute_metrics(pred_texts, true_texts)
    print(f"测试4 - 完全错误: WRR={WRR:.1f}%, WRR_IF={WRR_IF:.1f}% (期望: 0%, 0%)")
    
    # 测试用例5：边界情况 - 空列表
    WRR, WRR_IF, correct, correct_one_error = compute_metrics([], [])
    print(f"测试5 - 空列表: WRR={WRR:.1f}%, WRR_IF={WRR_IF:.1f}% (期望: 0%, 0%)")

if __name__ == '__main__':
    test_metrics() 