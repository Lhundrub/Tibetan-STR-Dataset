import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=2):
        super().__init__()
        
        # CNN特征提取器（简化版）
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # H/2, W/2
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/4, W/2
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8, W/2
        )
        
        # RNN层
        self.feature_height = 16  # CNN后的特征高度
        self.rnn_input_dim = 256 * self.feature_height
        self.hidden_dim = hidden_dim
        
        # 投影层
        self.fc_proj = nn.Linear(self.rnn_input_dim, hidden_dim)
        
        # LSTM层（双向）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # 双向所以除以2
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, src):
        # src: [B, 1, H, W]
        features = self.cnn(src)  # [B, 256, H/8, W/2]
        b, c, h, w = features.size()
        
        # 拉平成序列 [B, W, C*H]
        features = features.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        features = self.fc_proj(features)  # [B, W, hidden_dim]
        
        # LSTM处理
        lstm_out, _ = self.lstm(features)  # [B, W, hidden_dim]
        
        # 输出分类
        logits = self.fc(lstm_out)  # [B, W, vocab_size]
        return logits.permute(1, 0, 2)  # [W, B, vocab_size] 适配CTC 