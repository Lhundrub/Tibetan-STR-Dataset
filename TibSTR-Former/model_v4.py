import torch
import torch.nn as nn
import torchvision.models as models

class CNNBiLSTMTransformer(nn.Module):
    """v4: 完整模型 - 残差CNN+FPN+注意力+BiLSTM+Transformer，所有模块协同工作"""
    def __init__(self, vocab_size, hidden_dim=256, nhead=8, num_layers=3, lstm_hidden=512, lstm_layers=2):
        super().__init__()

        # CNN特征提取部分 - 添加残差连接
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

        # 投影层参数
        self.hidden_dim = hidden_dim
        self.feature_height = 16  # CNN后的特征高度
        
        # CNN投影到LSTM输入
        self.cnn_proj = nn.Linear(512 * self.feature_height, lstm_hidden)
        
        # 双向LSTM层 - 增强序列建模能力
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden // 2,  # 双向LSTM，所以隐藏层大小减半
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        # LSTM投影到Transformer层
        self.lstm_proj = nn.Linear(lstm_hidden, hidden_dim)
        
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
        
        # Dropout用于防止过拟合
        self.dropout = nn.Dropout(0.2)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        
        # 额外初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
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
        
        # 确保特征尺寸正确
        b, c, h, w = features.size()
        
        # 确保高度与预期一致
        if h != self.feature_height:
            raise ValueError(f"特征高度 {h} 与预期的 {self.feature_height} 不匹配")
        
        # 2. 拉平成序列并投影: [B, W, 512*H]
        features = features.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        features = self.cnn_proj(features)  
        features = self.dropout(features)  # 防止过拟合
        
        # 3. 双向LSTM序列建模
        lstm_out, _ = self.lstm(features)  # [B, W, lstm_hidden]
        lstm_out = self.dropout(lstm_out)
        
        # 添加LSTM层的残差连接 (如果维度匹配)
        if features.size(-1) == lstm_out.size(-1):
            lstm_out = lstm_out + features  # 残差连接
        
        # 4. 投影到Transformer尺寸
        transformer_in = self.lstm_proj(lstm_out)  # [B, W, hidden_dim]
        
        # 5. 应用自注意力增强序列特征
        transformer_in = self.self_attention(transformer_in)
        
        # 6. Transformer编码
        transformer_in = self.pos_encoder(transformer_in)
        output = self.transformer_encoder(transformer_in)
        output = self.dropout(output)
        
        # 7. 最终投影到词表大小
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