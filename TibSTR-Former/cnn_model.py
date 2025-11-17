import torch
import torch.nn as nn

class PureCNNOCR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # CNN特征提取层
        self.cnn_features = nn.Sequential(
            # 第一层：输入为灰度图像 [1, H, W]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 高度/2, 宽度/2
            
            # 第二层
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 高度/4, 宽度/2
            
            # 第三层
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 双层卷积增加特征提取能力
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 高度/8, 宽度/2
            
            # 第四层
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 双层卷积
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 高度/16, 宽度/2
        )
        
        # 高度压缩模块 - 自适应将任何高度压缩到1
        self.height_compress = nn.AdaptiveAvgPool2d((1, None))
        
        # 全连接层，将特征映射到字符类别
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Conv2d(512, vocab_size, kernel_size=1)
    
    def forward(self, x):
        # 输入: [B, 1, H, W] - 支持任意高度和宽度
        features = self.cnn_features(x)  # [B, 512, H/16, W/2]
        
        # 自适应压缩高度为1
        features = self.height_compress(features)  # [B, 512, 1, W/2]
        
        # 应用分类器
        features = self.dropout(features)
        logits = self.classifier(features)  # [B, vocab_size, 1, W/2]
        
        # 调整维度以适配CTC损失: [W, B, C]
        logits = logits.squeeze(2)  # 移除高度维度: [B, vocab_size, W/2]
        logits = logits.permute(2, 0, 1)  # [W/2, B, vocab_size]
        
        return logits

class DeepCNNOCR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # 第一阶段
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 高度/2, 宽度/2
        )
        
        # 第二阶段
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 高度/4, 宽度/4
        )
        
        # 第三阶段
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 高度/8, 宽度/4
        )
        
        # 第四阶段
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)  # 保持高度/8, 宽度/4
        )
        
        # 高度自适应压缩
        self.height_compress = nn.AdaptiveAvgPool2d((1, None))
        
        # 最终分类器
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Conv2d(512, vocab_size, kernel_size=1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入: [B, 1, H, W] - 自适应任意尺寸
        x = self.stage1(x)  # [B, 64, H/2, W/2]
        x = self.stage2(x)  # [B, 128, H/4, W/4]
        x = self.stage3(x)  # [B, 256, H/8, W/4]
        x = self.stage4(x)  # [B, 512, H/8, W/4]
        
        # 自适应压缩高度到1
        x = self.height_compress(x)  # [B, 512, 1, W/4]
        
        x = self.dropout(x)
        logits = self.classifier(x)  # [B, vocab_size, 1, W/4]
        logits = logits.squeeze(2)  # [B, vocab_size, W/4]
        
        # 调整维度以适配CTC损失: [W, B, C]
        logits = logits.permute(2, 0, 1)  # [W/4, B, vocab_size]
        
        return logits 