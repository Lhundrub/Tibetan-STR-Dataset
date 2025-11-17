import torch
import torch.nn as nn
import torchvision.models as models

class CNNTransformerBase(nn.Module):
    """v0: 基础CNN+Transformer - 移除残差连接的基线模型"""
    def __init__(self, vocab_size, hidden_dim=256, nhead=8, num_layers=3):
        super().__init__()

        # 基础CNN架构 - 无残差连接
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # H/2, W/2
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/4, W/2
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/8, W/2
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 序列投影层
        self.hidden_dim = hidden_dim
        self.feature_height = 16  # CNN后的特征高度
        self.fc_proj = nn.Linear(512 * self.feature_height, hidden_dim)

        # Transformer编码器
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # 基础CNN特征提取（无残差连接）
        x = self.conv1(src)
        x = self.pool1(x)
        
        x = self.conv2(x)  # 直接堆叠，无残差连接
        x = self.pool2(x)
        
        x = self.conv3(x)  # 直接堆叠，无残差连接
        
        x = self.conv4(x)
        x = self.pool3(x)
        
        features = self.conv5(x)  # 直接堆叠，无残差连接
        
        # 特征序列化
        b, c, h, w = features.size()
        if h != self.feature_height:
            raise ValueError(f"特征高度 {h} 与预期的 {self.feature_height} 不匹配")
        
        features = features.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        features = self.fc_proj(features)
        features = self.pos_encoder(features)
        output = self.transformer_encoder(features)
        logits = self.fc(output)
        return logits.permute(1, 0, 2)  # [W, B, vocab_size] 适配CTC

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x) 