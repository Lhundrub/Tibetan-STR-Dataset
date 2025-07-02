import torch
import torch.nn as nn
from torchvision import transforms
from model import CNNTransformer
from data_loader import OCRDataset, collate_fn
from torch.utils.data import DataLoader

def debug_model():
    # 加载词汇表
    with open('tibetan_vocab_full.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    vocab_size = len(vocab)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTransformer(vocab_size).to(device)
    
    # 创建测试输入
    test_input = torch.randn(1, 1, 32, 128).to(device)
    print(f"输入形状: {test_input.shape}")
    
    # 逐步检查模型各部分
    model.eval()
    with torch.no_grad():
        # CNN特征提取
        features = model.cnn(test_input)
        print(f"CNN输出形状: {features.shape}")
        
        b, c, h, w = features.size()
        print(f"特征维度: batch={b}, channels={c}, height={h}, width={w}")
        
        # 特征重塑
        feature_dim = c * h
        print(f"特征维度 (c*h): {feature_dim}")
        
        # 检查投影层是否正确初始化
        if model.fc_proj is None:
            model._init_projection_layer(feature_dim)
        
        # 重塑特征
        features_reshaped = features.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        print(f"重塑后特征形状: {features_reshaped.shape}")
        
        # 投影
        features_proj = model.fc_proj(features_reshaped)
        print(f"投影后特征形状: {features_proj.shape}")
        
        # 位置编码
        features_pos = model.pos_encoder(features_proj)
        print(f"位置编码后形状: {features_pos.shape}")
        
        # Transformer编码
        transformer_out = model.transformer_encoder(features_pos)
        print(f"Transformer输出形状: {transformer_out.shape}")
        
        # 最终输出
        logits = model.fc(transformer_out)
        print(f"最终logits形状: {logits.shape}")
        
        # 转置为CTC格式
        logits_ctc = logits.permute(1, 0, 2)
        print(f"CTC格式logits形状: {logits_ctc.shape}")
        
        print(f"\n序列长度分析:")
        print(f"  输入图像宽度: {test_input.shape[3]}")
        print(f"  CNN下采样后宽度: {w}")
        print(f"  最终序列长度: {logits_ctc.shape[0]}")
        print(f"  下采样比例: {test_input.shape[3] / w:.2f}")
        
        print(f"\n问题分析:")
        print(f"  序列长度太短！最长文本有28个字符，但模型只能输出8个时间步")
        print(f"  这会导致CTC无法正确对齐长文本")
        
        # 测试不同输入宽度
        print(f"\n测试不同输入宽度:")
        for width in [256, 512, 1024]:
            test_wide = torch.randn(1, 1, 32, width).to(device)
            features_wide = model.cnn(test_wide)
            _, _, _, w_wide = features_wide.size()
            print(f"  输入宽度{width} -> 输出宽度{w_wide} (下采样比例: {width/w_wide:.1f})")

if __name__ == '__main__':
    debug_model() 