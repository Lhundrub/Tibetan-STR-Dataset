import torch
import torch.nn as nn
import torchvision.models as models

class CNNTransformerAttention(nn.Module):
    """v3: 残差CNN+FPN+注意力+Transformer - 去除BiLSTM的版本，影响序列建模能力"""
    def __init__(self, vocab_size, hidden_dim=256, nhead=8, num_layers=3):
        super().__init__()

        # 残差CNN特征提取主干
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # H/2, W/2
        
        # 第二层 - 残差块
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )
        self.downsample1 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/4, W/2
        
        # 第三层 - 残差块
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )
        self.downsample2 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 第三层2 - 残差块
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        self.downsample3 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/8, W/2
        
        # 第四层 - 残差块
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        self.relu4 = nn.ReLU(inplace=True)
        
        # 特征金字塔网络（FPN）
        self.lateral_conv4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(128, 256, kernel_size=1)
        
        self.fpn_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # FPN输出特征融合
        self.fusion_conv1 = nn.Conv2d(256 * 3, 512, kernel_size=3, padding=1)
        self.fusion_bn1 = nn.BatchNorm2d(512)
        self.fusion_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fusion_bn2 = nn.BatchNorm2d(512)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # 多维注意力机制
        self.channel_attention = ChannelAttention(512)
        self.spatial_attention = SpatialAttention()

        # 序列投影层
        self.hidden_dim = hidden_dim
        self.feature_height = 16  # CNN后的特征高度
        self.fc_proj = nn.Linear(512 * self.feature_height, hidden_dim)
        
        # 自注意力层用于序列建模前的特征增强
        self.self_attention = SelfAttention(hidden_dim)

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
        
    def _upsample_add(self, x, y):
        """上采样x并与y相加"""
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, src):
        # 第一层卷积
        x = self.conv1(src)
        x = self.pool1(x)
        
        # 第二层残差块
        identity = self.downsample1(x)
        x = self.conv2_1(x)
        x = x + identity
        x = self.relu2(x)
        c2 = x
        x = self.pool2(x)
        
        # 第三层残差块
        identity = self.downsample2(x)
        x = self.conv3_1(x)
        x = x + identity
        x = self.relu3(x)
        c3 = x
        
        # 第三层第二部分残差块
        identity = self.downsample3(x)
        x = self.conv3_2(x)
        x = x + identity
        x = self.relu3_2(x)
        x = self.pool3(x)
        
        # 第四层残差块
        identity = x
        x = self.conv4_1(x)
        x = x + identity
        c4 = self.relu4(x)
        
        # FPN自顶向下路径
        p4 = self.lateral_conv4(c4)
        p3 = self._upsample_add(p4, self.lateral_conv3(c3))
        p2 = self._upsample_add(p3, self.lateral_conv2(c2))
        
        # FPN最终特征
        p4 = self.fpn_conv4(p4)
        p3 = self.fpn_conv3(p3)
        p2 = self.fpn_conv2(p2)
        
        # 统一特征尺寸到p4的大小
        p3_resized = nn.functional.interpolate(p3, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p2_resized = nn.functional.interpolate(p2, size=p4.shape[2:], mode='bilinear', align_corners=False)
        
        # 特征融合
        multi_scale_features = torch.cat([p4, p3_resized, p2_resized], dim=1)
        x = self.fusion_conv1(multi_scale_features)
        x = self.fusion_bn1(x)
        x = self.fusion_relu(x)
        x = self.fusion_conv2(x)
        x = self.fusion_bn2(x)
        features = self.fusion_relu(x)
        
        # 应用多维注意力机制
        att_ch = self.channel_attention(features)
        features = att_ch * features
        
        att_sp = self.spatial_attention(features)
        features = att_sp * features
        
        # 特征序列化
        b, c, h, w = features.size()
        if h != self.feature_height:
            raise ValueError(f"特征高度 {h} 与预期的 {self.feature_height} 不匹配")
        
        features = features.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        features = self.fc_proj(features)  # [B, W, hidden_dim]
        
        # 应用自注意力增强序列特征
        features = self.self_attention(features)
        
        # Transformer编码
        features = self.pos_encoder(features)
        output = self.transformer_encoder(features)
        logits = self.fc(output)  # [B, W, vocab_size]
        
        return logits.permute(1, 0, 2)  # [W, B, vocab_size] 适配CTC

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [B, L, D]
        batch_size, seq_len, hidden_dim = x.size()
        
        # 生成Q, K, V投影
        q = self.query(x)  # [B, L, D]
        k = self.key(x)    # [B, L, D] 
        v = self.value(x)  # [B, L, D]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(1, 2))  # [B, L, L]
        scores = scores / self.scale
        
        # 应用softmax获得注意力权重
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权聚合值向量
        output = torch.matmul(attention_weights, v)  # [B, L, D]
        
        return output + x  # 残差连接

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