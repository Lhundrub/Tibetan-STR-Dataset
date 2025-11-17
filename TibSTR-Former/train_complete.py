import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from lightning.fabric import Fabric
from torch.optim import Adam
import argparse
import os
import glob
from datetime import datetime
import numpy as np
from typing import List, Tuple

from model_v0 import CNNTransformerBase
from model_v1 import CNNTransformerResidual
from model_v2 import CNNTransformerFPN
from model_v3 import CNNTransformerAttention
from model_v4 import CNNBiLSTMTransformer
from data_loader import OCRDataset, collate_fn
from ocr_transforms import get_ocr_transforms, get_light_augmentation, get_heavy_augmentation

MODEL_DICT = {
    'v0': CNNTransformerBase,
    'v1': CNNTransformerResidual,
    'v2': CNNTransformerFPN,
    'v3': CNNTransformerAttention,
    'v4': CNNBiLSTMTransformer
}

MODEL_DESCRIPTIONS = {
    'v0': 'CNN + Transformer',
    'v1': 'CNN + Residual + Transformer',
    'v2': 'CNN + FPN + Transformer',
    'v3': 'CNN + FPN + Attention + Transformer',
    'v4': 'CNN + FPN + BiLSTM + Transformer'
}

torch.set_float32_matmul_precision('high')

def compute_loss(logits, targets, target_lengths, blank=0):
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    input_lengths = torch.full(
        size=(logits.size(1),),
        fill_value=logits.size(0),
        dtype=torch.long,
        device=log_probs.device
    )
    
    targets_concat = []
    if not isinstance(target_lengths, torch.Tensor):
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=log_probs.device)
    
    for i, length in enumerate(target_lengths):
        target_seq = targets[i][:length]
        targets_concat.extend(target_seq.tolist())
    
    targets_concat = torch.tensor(targets_concat, dtype=torch.long, device=log_probs.device)
    actual_target_lengths = target_lengths.clone()
    
    loss = nn.CTCLoss(blank=blank, zero_infinity=True)(
        log_probs, targets_concat, input_lengths, actual_target_lengths
    )
    return loss


def decode_predictions(preds, idx2char, blank=0, remove_tokens=None):
    if remove_tokens is None:
        remove_tokens = ['<sos>', '<eos>', '<pad>', '<unk>']
    
    decoded_texts = []
    if preds.ndim == 3:
        preds = preds.argmax(dim=2)
    preds = preds.cpu().numpy()
    
    for i in range(preds.shape[1]):
        raw_pred = preds[:, i]
        decoded = []
        prev_char_idx = None
        
        for char_idx in raw_pred:
            if char_idx != blank and char_idx != prev_char_idx:
                char = idx2char.get(char_idx, '')
                if char not in remove_tokens:
                    decoded.append(char)
            prev_char_idx = char_idx
        
        decoded_texts.append(''.join(decoded))
    return decoded_texts


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


def compute_edit_distance_details(pred_text: str, true_text: str) -> Tuple[int, int, int, int]:
    dp = np.zeros((len(true_text) + 1, len(pred_text) + 1), dtype=np.int32)
    operations = np.zeros((len(true_text) + 1, len(pred_text) + 1), dtype=np.int32)
    
    for i in range(len(true_text) + 1):
        dp[i, 0] = i
        if i > 0:
            operations[i, 0] = 2
    
    for j in range(len(pred_text) + 1):
        dp[0, j] = j
        if j > 0:
            operations[0, j] = 1
    
    for i in range(1, len(true_text) + 1):
        for j in range(1, len(pred_text) + 1):
            if true_text[i-1] == pred_text[j-1]:
                dp[i, j] = dp[i-1, j-1]
                operations[i, j] = 0
            else:
                deletion = dp[i-1, j] + 1
                insertion = dp[i, j-1] + 1
                substitution = dp[i-1, j-1] + 1
                min_op = min(deletion, insertion, substitution)
                dp[i, j] = min_op
                
                if min_op == deletion:
                    operations[i, j] = 2
                elif min_op == insertion:
                    operations[i, j] = 1
                else:
                    operations[i, j] = 3
    
    i, j = len(true_text), len(pred_text)
    insertions, deletions, substitutions = 0, 0, 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and operations[i, j] == 0:
            i -= 1
            j -= 1
        elif j > 0 and operations[i, j] == 1:
            insertions += 1
            j -= 1
        elif i > 0 and operations[i, j] == 2:
            deletions += 1
            i -= 1
        elif i > 0 and j > 0 and operations[i, j] == 3:
            substitutions += 1
            i -= 1
            j -= 1
        else:
            if i > 0:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1
    
    return dp[len(true_text), len(pred_text)], insertions, deletions, substitutions


TIBETAN_DIACRITICS = {
    'ི', 'ུ', 'ེ', 'ོ', 'ཀ', 'ཁ', 'ག', 'ང', 'ཅ', 'ཆ', 'ཇ', 'ཉ', 
    'ཏ', 'ཐ', 'ད', 'ན', 'པ', 'ཕ', 'བ', 'མ', 'ཙ', 'ཚ', 'ཛ', 'ཝ', 
    'ཞ', 'ཟ', 'འ', 'ཡ', 'ར', 'ལ', 'ཤ', 'ས', 'ཧ', 'ཨ', '༡', '༢', 
    '༣', '༤', '༥', '༦', '༧', '༨', '༩', '༠', '༼', '༽', '།', '༄༅།།', 
    '༈', '་', '༜', '༄', 'ྱ', 'ྲ', 'ྀ', 'ཽ', 'ཻ', 'ྭ', 'ཾ', 'ླ', 
    'ཥ', 'ཋ', 'ཌ', 'ཊ'
}


def compute_crr(pred_texts: List[str], true_texts: List[str]) -> Tuple[float, int, int]:
    if len(pred_texts) != len(true_texts):
        raise ValueError("Length mismatch")
    
    total_chars = 0
    total_edit_distance = 0
    
    for pred, true in zip(pred_texts, true_texts):
        total_chars += len(true)
        total_edit_distance += edit_distance(pred, true)
    
    if total_chars == 0:
        return 0.0, 0, 0
    
    correct_chars = max(0, total_chars - total_edit_distance)
    CRR = (correct_chars / total_chars) * 100
    
    return CRR, total_chars, correct_chars


def compute_paper_CER(pred_texts: List[str], true_texts: List[str]) -> Tuple[float, int, int, int, int]:
    total_chars = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    
    for pred, true in zip(pred_texts, true_texts):
        total_chars += len(true)
        _, ins, dels, subs = compute_edit_distance_details(pred, true)
        total_substitutions += subs
        total_deletions += dels
        total_insertions += ins
    
    if total_chars == 0:
        return 0.0, 0, 0, 0, 0
    
    CER = ((total_substitutions + total_deletions + total_insertions) / total_chars) * 100
    
    return CER, total_substitutions, total_deletions, total_insertions, total_chars


def compute_IER(pred_texts: List[str], true_texts: List[str]) -> Tuple[float, int, int]:
    total_images = len(true_texts)
    incorrect_images = sum(1 for pred, true in zip(pred_texts, true_texts) if pred != true)
    
    if total_images == 0:
        return 0.0, 0, 0
    
    IER = (incorrect_images / total_images) * 100
    return IER, incorrect_images, total_images


def compute_DER(pred_texts: List[str], true_texts: List[str]) -> Tuple[float, int, int]:
    total_diacritics = 0
    total_diacritic_errors = 0
    
    for pred, true in zip(pred_texts, true_texts):
        true_diacritics = [c for c in true if c in TIBETAN_DIACRITICS]
        pred_diacritics = [c for c in pred if c in TIBETAN_DIACRITICS]
        
        total_diacritics += len(true_diacritics)
        
        if len(true_diacritics) == 0:
            total_diacritic_errors += len(pred_diacritics)
        elif len(pred_diacritics) == 0:
            total_diacritic_errors += len(true_diacritics)
        else:
            true_dia_str = ''.join(true_diacritics)
            pred_dia_str = ''.join(pred_diacritics)
            dia_edit_dist = edit_distance(pred_dia_str, true_dia_str)
            total_diacritic_errors += dia_edit_dist
    
    if total_diacritics == 0:
        return 0.0, 0, 0
    
    DER = (total_diacritic_errors / total_diacritics) * 100
    return DER, total_diacritic_errors, total_diacritics


def compute_SER(pred_texts: List[str], true_texts: List[str]) -> Tuple[float, int, int]:
    total_sentences = len(true_texts)
    incorrect_sentences = sum(1 for pred, true in zip(pred_texts, true_texts) if pred != true)
    
    if total_sentences == 0:
        return 0.0, 0, 0
    
    SER = (incorrect_sentences / total_sentences) * 100
    return SER, incorrect_sentences, total_sentences


def compute_all_metrics(pred_texts: List[str], true_texts: List[str]) -> dict:
    CRR, total_chars, correct_chars = compute_crr(pred_texts, true_texts)
    CER, S, D, I, N = compute_paper_CER(pred_texts, true_texts)
    IER, incorrect_imgs, total_imgs = compute_IER(pred_texts, true_texts)
    DER, incorrect_dia, total_dia = compute_DER(pred_texts, true_texts)
    SER, incorrect_sent, total_sent = compute_SER(pred_texts, true_texts)
    
    return {
        'CRR': CRR, 'CER': CER, 'IER': IER, 'DER': DER, 'SER': SER,
        'CER_substitutions': S, 'CER_deletions': D, 'CER_insertions': I,
        'correct_chars': correct_chars, 'total_chars': total_chars,
        'incorrect_images': incorrect_imgs, 'total_images': total_imgs,
        'incorrect_diacritics': incorrect_dia, 'total_diacritics': total_dia,
        'incorrect_sentences': incorrect_sent, 'total_sentences': total_sent,
    }


def train_model(model_version='v4', augmentation='medium', epochs=100, batch_size=16, lr=0.0001):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    fabric = Fabric(accelerator="cuda", devices=1, precision="16-mixed")
    fabric.launch()
    
    vocab_path = 'tibetan_vocab_optimized.txt'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    
    model_class = MODEL_DICT.get(model_version, CNNBiLSTMTransformer)
    model_desc = MODEL_DESCRIPTIONS.get(model_version, 'Unknown')
    
    fabric.print(f"\n{'='*60}")
    fabric.print(f"Tibetan OCR Training - Model {model_version.upper()}")
    fabric.print(f"{'='*60}")
    fabric.print(f"Model: {model_desc}")
    fabric.print(f"Vocab Size: {vocab_size}")
    fabric.print(f"Augmentation: {augmentation}")
    fabric.print(f"Batch Size: {batch_size}")
    fabric.print(f"Epochs: {epochs}")
    fabric.print(f"Learning Rate: {lr}")
    fabric.print(f"Precision: 16-bit mixed")
    fabric.print(f"{'='*60}\n")
    
    model = model_class(vocab_size)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model, optimizer = fabric.setup(model, optimizer)
    
    os.makedirs('saved_models', exist_ok=True)
    model_dir = f'saved_models/{model_version}'
    os.makedirs(model_dir, exist_ok=True)
    model_files = glob.glob(f"{model_dir}/best_model*.pth")
    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        fabric.print(f"Found checkpoint: {latest_model}")
        try:
            state_dict = fabric.load(latest_model)
            model.load_state_dict(state_dict)
            fabric.print("Loaded checkpoint successfully")
        except Exception as e:
            fabric.print(f"Failed to load: {e}")
            fabric.print("Training from scratch")
    
    transform_map = {
        'none': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'light': get_light_augmentation(),
        'medium': get_ocr_transforms(train=True, img_height=128, img_width=512),
        'heavy': get_heavy_augmentation()
    }
    
    train_transform = transform_map.get(augmentation, get_ocr_transforms(train=True))
    val_transform = get_ocr_transforms(train=False)
    
    train_dataset = OCRDataset('data/train_new.json', vocab_path, transform=train_transform)
    val_dataset = OCRDataset('data/val_new.json', vocab_path, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, persistent_workers=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=1, persistent_workers=False, pin_memory=True)
    
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)
    
    fabric.print(f"Train samples: {len(train_dataset)}")
    fabric.print(f"Val samples: {len(val_dataset)}\n")
    
    best_val_wrr = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (images, texts, text_lengths, raw_texts) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(images)
            loss = compute_loss(logits, texts, text_lengths, blank=0)
            
            if torch.isnan(loss) or torch.isinf(loss):
                fabric.print(f"Invalid loss: {loss.item()}")
                continue
            
            fabric.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % 30 == 0:
                avg_loss = total_loss / num_batches
                fabric.print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            val_wrr = validate_model(fabric, model, val_loader, idx2char, epoch + 1)
            
            if val_wrr > best_val_wrr:
                best_val_wrr = val_wrr
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f'{model_dir}/best_model_{timestamp}_epoch{epoch+1}.pth'
                fabric.save(model_filename, model.state_dict())
                fabric.print(f'Saved best model: CRR={val_wrr:.2f}%\n')
        else:
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            fabric.print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}')
    
    fabric.print("\nTraining completed!")


def validate_model(fabric, model, val_loader, idx2char, epoch):
    model.eval()
    all_preds = []
    all_true = []
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, texts, text_lengths, raw_texts in val_loader:
            logits = model(images)
            loss = compute_loss(logits, texts, text_lengths, blank=0)
            val_loss += loss.item()
            val_batches += 1
            
            preds = decode_predictions(logits, idx2char, blank=0)
            all_preds.extend(preds)
            all_true.extend(raw_texts)
    
    metrics = compute_all_metrics(all_preds, all_true)
    
    fabric.print(f'\n{"="*60}')
    fabric.print(f'Epoch {epoch} Validation Results')
    fabric.print(f'{"="*60}')
    fabric.print(f'Val Loss: {val_loss/val_batches:.4f}')
    fabric.print(f'\nMetrics:')
    fabric.print(f'  CRR: {metrics["CRR"]:.2f}% ({metrics["correct_chars"]}/{metrics["total_chars"]})')
    fabric.print(f'  CER: {metrics["CER"]:.2f}% (S={metrics["CER_substitutions"]}, D={metrics["CER_deletions"]}, I={metrics["CER_insertions"]})')
    fabric.print(f'  IER: {metrics["IER"]:.2f}% ({metrics["incorrect_images"]}/{metrics["total_images"]})')
    fabric.print(f'  DER: {metrics["DER"]:.2f}% ({metrics["incorrect_diacritics"]}/{metrics["total_diacritics"]})')
    fabric.print(f'  SER: {metrics["SER"]:.2f}% ({metrics["incorrect_sentences"]}/{metrics["total_sentences"]})')
    
    fabric.print(f'\nPrediction samples:')
    for i in range(min(3, len(all_preds))):
        match = "✓" if all_preds[i] == all_true[i] else "✗"
        fabric.print(f'  True: "{all_true[i]}"')
        fabric.print(f'  Pred: "{all_preds[i]}" {match}')
    fabric.print(f'{"="*60}\n')
    
    return metrics['CRR']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=['v0', 'v1', 'v2', 'v3', 'v4'], 
                       default='v4', help='Model version')
    parser.add_argument('--augmentation', '-a', choices=['none', 'light', 'medium', 'heavy'], 
                       default='medium', help='Data augmentation')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    model_desc = MODEL_DESCRIPTIONS.get(args.model, 'Unknown')
    
    print(f"\n{'='*60}")
    print(f"{'Tibetan OCR Training':^60}")
    print(f"{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"Description: {model_desc}")
    print(f"Precision: 16-bit mixed")
    print(f"Augmentation: {args.augmentation}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"{'='*60}\n")
    
    train_model(
        model_version=args.model,
        augmentation=args.augmentation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
