import torch
import torch.nn as nn

class SimpleCNNModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # 纯CNN架构，保持序列维度
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # H/2, W/2
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/4, W/2
            
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8, W/2
            
            # 第四层
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 不再池化，保持宽度
        )
        
        # 计算特征高度
        self.feature_height = 16  # 输入128高度 -> 128/8 = 16
        
        # 分类层 - 直接从CNN特征到词汇表
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(self.feature_height, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(512, vocab_size, kernel_size=1, stride=1, padding=0)
        )
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, src):
        # src: [B, 1, H, W]
        features = self.features(src)  # [B, 256, H/8, W/2]
        b, c, h, w = features.size()
        
        # 确保高度匹配
        if h != self.feature_height:
            raise ValueError(f"特征高度 {h} 与预期的 {self.feature_height} 不匹配")
        
        # 通过分类层：将高度维度压缩为1
        logits = self.classifier(features)  # [B, vocab_size, 1, W/2]
        
        # 重塑为序列格式
        logits = logits.squeeze(2)  # [B, vocab_size, W/2]
        logits = logits.permute(2, 0, 1)  # [W/2, B, vocab_size] 适配CTC
        
        return logits 