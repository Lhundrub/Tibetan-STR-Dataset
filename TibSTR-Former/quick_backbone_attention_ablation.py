import os
import argparse
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

from data_loader import OCRDataset, collate_fn
from ocr_transforms import get_ocr_transforms
from model_v4 import (
    CNNBiLSTMTransformer as V4Model,
    ChannelAttention as V4ChannelAttention,
    SpatialAttention as V4SpatialAttention,
)


def compute_loss(logits, targets, target_lengths, blank=0):
    log_probs = F.log_softmax(logits, dim=2)
    input_lengths = torch.full(
        size=(logits.size(1),),
        fill_value=logits.size(0),
        dtype=torch.long,
        device=log_probs.device,
    )
    if not isinstance(target_lengths, torch.Tensor):
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=log_probs.device)

    targets_concat = []
    for i, length in enumerate(target_lengths):
        targets_concat.extend(targets[i][:length].tolist())
    targets_concat = torch.tensor(targets_concat, dtype=torch.long, device=log_probs.device)

    loss = nn.CTCLoss(blank=blank, zero_infinity=True)(
        log_probs, targets_concat, input_lengths, target_lengths
    )
    return loss


def edit_distance(s1: str, s2: str) -> int:
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


def ctc_decode(preds: torch.Tensor, idx2char: Dict[int, str], blank=0) -> List[str]:
    if preds.ndim == 3:
        preds = preds.argmax(dim=2)
    preds = preds.detach().cpu().numpy()
    decoded = []
    for i in range(preds.shape[1]):
        raw = preds[:, i]
        out = []
        prev = None
        for ch in raw:
            if ch != blank and ch != prev:
                out.append(idx2char.get(int(ch), ''))
            prev = ch
        decoded.append(''.join(out))
    return decoded


def compute_crr(pred_texts: List[str], true_texts: List[str]) -> float:
    total_chars = 0
    total_edit = 0
    for p, t in zip(pred_texts, true_texts):
        total_chars += len(t)
        total_edit += edit_distance(p, t)
    if total_chars == 0:
        return 0.0
    correct = max(0, total_chars - total_edit)
    return (correct / total_chars) * 100.0


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class ResNetCRNN(nn.Module):
    """
    轻量对比模型：ResNet34/50 backbone (+ 可选CBAM) + BiLSTM + Transformer + CTC 头。
    仅用于快速消融，非最终SOTA结构。
    """

    def __init__(self, vocab_size: int, backbone: str = 'resnet34', use_cbam: bool = True,
                 hidden_dim: int = 256, lstm_hidden: int = 512, lstm_layers: int = 1,
                 trans_layers: int = 1, nhead: int = 8):
        super().__init__()

        if backbone == 'resnet50':
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            out_channels = 2048
        else:
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            out_channels = 512

        # 改为接收单通道输入（用RGB权重均值初始化）
        w = net.conv1.weight.data
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data = w.mean(dim=1, keepdim=True)
        net.conv1 = new_conv1

        # 取至 layer4 的特征
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )

        # 通道降维到 512
        self.conv_reduce = nn.Conv2d(out_channels, 512, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.use_cbam = use_cbam
        if use_cbam:
            self.ca = ChannelAttention(512)
            self.sa = SpatialAttention()

        self.lstm_in = nn.Linear(512 * 4, lstm_hidden)  # 估计特征高度≈4（输入H=128时）
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0,
        )
        self.proj = nn.Linear(lstm_hidden, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [B,1,128,W]
        f = self.backbone(x)  # [B,C',H',W'] 预期H'≈4, W'≈W/32
        f = self.relu(self.bn_reduce(self.conv_reduce(f)))  # [B,512,H',W']
        if self.use_cbam:
            f = self.ca(f) * f
            f = self.sa(f) * f

        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)  # [B,W,512*H']
        f = self.lstm_in(f)
        f, _ = self.lstm(f)
        f = self.dropout(f)
        f = self.proj(f)
        f = self.trans(f)
        logits = self.fc(f)  # [B,W,V]
        return logits.permute(1, 0, 2)  # [W,B,V]


# ---- V4 attention variants (monkey-patch without touching model_v4.py) ----

class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, max(8, channels // reduction), 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(max(8, channels // reduction), channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.relu(self.conv1(w))
        w = self.sigmoid(self.conv2(w))
        return w


class ECAAttention(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B,C,1,1]
        y = y.squeeze(-1).transpose(1, 2)  # [B,1,C]
        y = self.conv1d(y)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # [B,C,1,1]
        return y


class IdentityChannel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

    def forward(self, x):
        return torch.ones((x.size(0), x.size(1), 1, 1), device=x.device, dtype=x.dtype)


class IdentitySpatial(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)


def build_v4_with_attention(vocab_size: int, attn_type: str = 'cbam') -> nn.Module:
    model = V4Model(vocab_size)
    # replace attentions according to attn_type
    if attn_type.lower() == 'cbam':
        # keep original
        return model
    elif attn_type.lower() == 'se':
        model.channel_attention = SEAttention(512)
        model.spatial_attention = IdentitySpatial()
    elif attn_type.lower() == 'eca':
        model.channel_attention = ECAAttention(512, k_size=3)
        model.spatial_attention = IdentitySpatial()
    elif attn_type.lower() == 'channel':
        model.channel_attention = V4ChannelAttention(512)
        model.spatial_attention = IdentitySpatial()
    elif attn_type.lower() == 'spatial':
        model.channel_attention = IdentityChannel(512)
        model.spatial_attention = V4SpatialAttention()
    elif attn_type.lower() == 'none':
        model.channel_attention = IdentityChannel(512)
        model.spatial_attention = IdentitySpatial()
    else:
        raise ValueError(f'Unknown attn_type: {attn_type}')
    return model


def moving_average(y: List[float], k: int = 3) -> List[float]:
    if k <= 1 or len(y) == 0:
        return y
    out = []
    for i in range(len(y)):
        s = max(0, i - k + 1)
        out.append(float(np.mean(y[s:i + 1])))
    return out


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def train_one(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for images, texts, text_lengths, _ in loader:
        images = images.to(device)
        texts = texts.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = compute_loss(logits, texts, text_lengths, blank=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, idx2char, device) -> float:
    model.eval()
    preds_all, trues_all = [], []
    for images, texts, text_lengths, raw_texts in loader:
        images = images.to(device)
        logits = model(images)
        preds = ctc_decode(logits, idx2char, blank=0)
        preds_all.extend(preds)
        trues_all.extend(list(raw_texts))
    return compute_crr(preds_all, trues_all)


def main():
    parser = argparse.ArgumentParser(description='Quick Ablation: Backbone (ResNet) and V4 Attention Variants')
    parser.add_argument('--data-train', default='data/train_new.json')
    parser.add_argument('--data-val', default='data/val_new.json')
    parser.add_argument('--vocab', default='tibetan_vocab_optimized.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=16)
    # 移除子集裁剪，始终使用全量数据（与主实验一致）
    parser.add_argument('--output-dir', default='quick_ablation')
    parser.add_argument('--backbone', choices=['resnet34', 'resnet50'], default='resnet34')
    parser.add_argument('--backbones', default='resnet34,resnet50', help='comma separated list for summary figure')
    parser.add_argument('--exp', choices=['backbone', 'v4_attention', 'both'], default='both')
    parser.add_argument('--v4-attn-list', default='cbam,se,eca,channel,spatial,none',
                        help='comma separated attention variants for V4')
    parser.add_argument('--freeze-v4-cnn', action='store_true', help='freeze CNN+FPN of V4 for quick fine-tuning')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with open(args.vocab, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    idx2char = {i: ch for i, ch in enumerate(vocab)}
    vocab_size = len(vocab)

    # transforms
    train_tf = get_ocr_transforms(train=True, img_height=128, img_width=512)
    val_tf = get_ocr_transforms(train=False, img_height=128, img_width=512)

    # datasets (full)
    train_ds = OCRDataset(args.data_train, args.vocab, transform=train_tf)
    val_ds = OCRDataset(args.data_val, args.vocab, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, persistent_workers=False, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=1, persistent_workers=False, pin_memory=True)

    # Backbone ablation (ResNet)
    if args.exp in ('backbone', 'both'):
        # 训练各 ResNet 变体（仅 CBAM 版本用于总结），并绘制单独曲线
        variants = {
            f'{args.backbone}_no_cbam': False,
            f'{args.backbone}_cbam': True,
        }
        curves = {}
        for name, use_cbam in variants.items():
            model = ResNetCRNN(vocab_size, backbone=args.backbone, use_cbam=use_cbam).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            train_losses, val_crrs = [], []
            for ep in range(args.epochs):
                tl = train_one(model, train_loader, optimizer, device)
                crr = evaluate(model, val_loader, idx2char, device)
                train_losses.append(tl); val_crrs.append(crr)
                print(f'[{name}] Epoch {ep+1}/{args.epochs}  TrainLoss={tl:.4f}  ValCRR={crr:.2f}%')
            curves[name] = (train_losses, val_crrs)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for name, (losses, _) in curves.items():
            plt.plot(range(1, len(losses) + 1), moving_average(losses, 3), label=name)
        plt.xlabel('Epoch'); plt.ylabel('Train Loss'); plt.title(f'Training Loss (ResNet backbone: {args.backbone})'); plt.grid(True); plt.legend()
        plt.subplot(1, 2, 2)
        for name, (_, crrs) in curves.items():
            plt.plot(range(1, len(crrs) + 1), moving_average(crrs, 2), label=name)
        plt.xlabel('Epoch'); plt.ylabel('CRR (%)'); plt.title(f'Validation CRR (ResNet backbone: {args.backbone})'); plt.grid(True); plt.legend()
        out_path = os.path.join(args.output_dir, f'curves_backbone_{args.backbone}.png')
        plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()

        import json
        save_json = {}
        for name, (losses, crrs) in curves.items():
            save_json[name] = {'train_loss': [float(x) for x in losses], 'val_crr': [float(x) for x in crrs]}
        with open(os.path.join(args.output_dir, f'curves_backbone_{args.backbone}.json'), 'w', encoding='utf-8') as f:
            json.dump(save_json, f, indent=2, ensure_ascii=False)
        print(f'Backbone curves saved to: {out_path}')

    # 生成“图1：Ours(V4) vs ResNet-34/50（CBAM）”的汇总柱状图（取最佳CRR）
    backbones = [s.strip() for s in args.backbones.split(',') if s.strip()]
    if args.exp in ('backbone', 'both') and backbones:
        # 训练 V4 baseline（ours）
        v4_model = V4Model(vocab_size).to(device)
        v4_opt = torch.optim.AdamW(v4_model.parameters(), lr=1e-4, weight_decay=1e-5)
        v4_losses, v4_crrs = [], []
        for ep in range(args.epochs):
            tl = train_one(v4_model, train_loader, v4_opt, device)
            crr = evaluate(v4_model, val_loader, idx2char, device)
            v4_losses.append(tl); v4_crrs.append(crr)
            print(f'[v4_ours] Epoch {ep+1}/{args.epochs}  TrainLoss={tl:.4f}  ValCRR={crr:.2f}%')
        v4_params = count_params_m(v4_model)

        labels, bests = ['ours(V4)'], [max(v4_crrs) if v4_crrs else 0.0]
        res_curves = {}  # for combined curve plot
        for bb in backbones:
            model = ResNetCRNN(vocab_size, backbone=bb, use_cbam=True).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            losses, crrs = [], []
            for ep in range(args.epochs):
                tl = train_one(model, train_loader, opt, device)
                losses.append(tl)
                crrs.append(evaluate(model, val_loader, idx2char, device))
                print(f'[{bb}_cbam_summary] Epoch {ep+1}/{args.epochs}  TrainLoss={tl:.4f}  ValCRR={crrs[-1]:.2f}%')
            labels.append(bb.replace('resnet', 'resnet-'))
            bests.append(max(crrs) if crrs else 0.0)
            res_curves[bb] = (losses, crrs, count_params_m(model))

        plt.figure(figsize=(7, 5))
        x = np.arange(len(labels))
        plt.bar(x, bests, color=['#2ecc71', '#3498db', '#9b59b6'][:len(labels)], edgecolor='black', alpha=0.85)
        for i, v in enumerate(bests):
            plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
        plt.xticks(x, labels)
        plt.ylabel('Best CRR (%)')
        plt.title('Backbone Comparison (Best CRR)')
        out_sum1 = os.path.join(args.output_dir, 'summary_backbones.png')
        plt.tight_layout(); plt.savefig(out_sum1, dpi=300, bbox_inches='tight'); plt.close()
        print(f'Saved: {out_sum1}')

        # 组合平滑曲线（图1）：ours(V4) vs resnet-34_cbam vs resnet-50_cbam
        if len(backbones) >= 1:
            plt.figure(figsize=(12, 5))
            # Train loss
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(v4_losses) + 1), moving_average(v4_losses, 3),
                     label=f'ours(V4) [{v4_params:.1f}M]')
            for bb in backbones:
                if bb in res_curves:
                    losses, _, p = res_curves[bb]
            plt.plot(range(1, len(losses) + 1), moving_average(losses, 3),
                             label=f'{bb.replace("resnet","resnet-")} [{p:.1f}M]')
            plt.xlabel('Epoch'); plt.ylabel('Train Loss'); plt.title('Training Loss (only backbone changes)')
            plt.grid(True); plt.legend(fontsize=9)
            # Val CRR
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(v4_crrs) + 1), moving_average(v4_crrs, 2),
                     label=f'ours(V4) [{v4_params:.1f}M]')
            for bb in backbones:
                if bb in res_curves:
                    _, crrs, p = res_curves[bb]
                    plt.plot(range(1, len(crrs) + 1), moving_average(crrs, 2),
                             label=f'{bb.replace("resnet","resnet-")} [{p:.1f}M]')
            plt.xlabel('Epoch'); plt.ylabel('CRR (%)'); plt.title('Validation CRR (only backbone changes)')
            plt.grid(True); plt.legend(fontsize=9)
            fig1_path = os.path.join(args.output_dir, 'figure1_backbones_curves.png')
            plt.tight_layout(); plt.savefig(fig1_path, dpi=300, bbox_inches='tight'); plt.close()
            print(f'Saved: {fig1_path}')

    # V4 attention ablation (cbam/se/eca/channel/spatial/none)
    if args.exp in ('v4_attention', 'both'):
        attn_list = [s.strip() for s in args.v4_attn_list.split(',') if s.strip()]
        curves = {}
        for attn in attn_list:
            name = f'v4_{attn}'
            model = build_v4_with_attention(vocab_size, attn)
            if args.freeze_v4_cnn:
                for n, p in model.named_parameters():
                    if any(k in n for k in ['conv', 'bn', 'fpn', 'fusion', 'lateral']):
                        p.requires_grad = False
            model = model.to(device)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
            train_losses, val_crrs = [], []
            for ep in range(args.epochs):
                tl = train_one(model, train_loader, optimizer, device)
                crr = evaluate(model, val_loader, idx2char, device)
                train_losses.append(tl); val_crrs.append(crr)
                print(f'[{name}] Epoch {ep+1}/{args.epochs}  TrainLoss={tl:.4f}  ValCRR={crr:.2f}%')
            curves[name] = (train_losses, val_crrs)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for name, (losses, _) in curves.items():
            plt.plot(range(1, len(losses) + 1), moving_average(losses, 3), label=name)
        plt.xlabel('Epoch'); plt.ylabel('Train Loss'); plt.title('Training Loss (V4 attention variants)'); plt.grid(True); plt.legend(fontsize=8)
        plt.subplot(1, 2, 2)
        for name, (_, crrs) in curves.items():
            plt.plot(range(1, len(crrs) + 1), moving_average(crrs, 2), label=name)
        plt.xlabel('Epoch'); plt.ylabel('CRR (%)'); plt.title('Validation CRR (V4 attention variants)'); plt.grid(True); plt.legend(fontsize=8)
        out_path2 = os.path.join(args.output_dir, 'curves_v4_attention.png')
        plt.tight_layout(); plt.savefig(out_path2, dpi=300, bbox_inches='tight'); plt.close()

        import json
        save_json2 = {}
        for name, (losses, crrs) in curves.items():
            save_json2[name] = {'train_loss': [float(x) for x in losses], 'val_crr': [float(x) for x in crrs]}
        with open(os.path.join(args.output_dir, 'curves_v4_attention.json'), 'w', encoding='utf-8') as f:
            json.dump(save_json2, f, indent=2, ensure_ascii=False)
        print(f'V4 attention curves saved to: {out_path2}')

        # 生成“图2：Ours(CBAM) vs SE/ECA/Channel/Spatial”的汇总柱状图（取最佳CRR）
        # 仅保留四类：cbam / eca / channel / spatial，且命名不带前缀
        order = ['cbam', 'eca', 'channel', 'spatial']
        labels, bests = [], []
        fig2_loss_curves, fig2_crr_curves, fig2_params = [], [], []
        for k in order:
            name = f'v4_{k}'
            if name in curves:
                pretty = 'ours(CBAM)' if k == 'cbam' else (k.upper() if k in ['se', 'eca'] else k)
                labels.append(pretty)
                bests.append(max(curves[name][1]) if curves[name][1] else 0.0)
                m = build_v4_with_attention(vocab_size, k)
                fig2_params.append(count_params_m(m))
                fig2_loss_curves.append(curves[name][0])
                fig2_crr_curves.append(curves[name][1])

        # 组合平滑曲线（图2）：仅更换注意力，其他保持与V4一致
        if labels:
            plt.figure(figsize=(12, 5))
            # Train loss
            plt.subplot(1, 2, 1)
            for name, losses, p in zip(labels, fig2_loss_curves, fig2_params):
                plt.plot(range(1, len(losses) + 1), moving_average(losses, 3), label=f'{name} [{p:.1f}M]')
            plt.xlabel('Epoch'); plt.ylabel('Train Loss'); plt.title('Training Loss (only attention changes)')
            plt.grid(True); plt.legend(fontsize=9)
            # Val CRR
            plt.subplot(1, 2, 2)
            for name, crrs, p in zip(labels, fig2_crr_curves, fig2_params):
                plt.plot(range(1, len(crrs) + 1), moving_average(crrs, 2), label=f'{name} [{p:.1f}M]')
            plt.xlabel('Epoch'); plt.ylabel('CRR (%)'); plt.title('Validation CRR (only attention changes)')
            plt.grid(True); plt.legend(fontsize=9)
            fig2_path = os.path.join(args.output_dir, 'figure2_attention_curves.png')
            plt.tight_layout(); plt.savefig(fig2_path, dpi=300, bbox_inches='tight'); plt.close()
            print(f'Saved: {fig2_path}')


if __name__ == '__main__':
    main()
