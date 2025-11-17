import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
import random
import shutil
import os
from datetime import datetime
from typing import List, Tuple
import numpy as np

from model_v0 import CNNTransformerBase
from model_v1 import CNNTransformerResidual
from model_v2 import CNNTransformerFPN
from model_v3 import CNNTransformerAttention
from model_v4 import CNNBiLSTMTransformer
from data_loader import OCRDataset, collate_fn
from ocr_transforms import get_ocr_transforms

MODEL_DICT = {
    'v0': CNNTransformerBase,
    'v1': CNNTransformerResidual,
    'v2': CNNTransformerFPN,
    'v3': CNNTransformerAttention,
    'v4': CNNBiLSTMTransformer
}


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


def create_test_split(val_json='data/val_new.json', 
                      test_ratio=0.5,
                      output_val='data/val_new.json',
                      output_test='data/test_new.json',
                      seed=42):
    print("\n" + "="*70)
    print("Creating independent test set")
    print("="*70)
    
    if os.path.exists(output_test):
        print(f"\nError: Test set already exists: {output_test}")
        print("Delete or backup existing test set to recreate")
        return
    
    with open(val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"\nOriginal validation samples: {len(val_data)}")
    
    random.seed(seed)
    random.shuffle(val_data)
    
    split_idx = int(len(val_data) * (1 - test_ratio))
    new_val = val_data[:split_idx]
    new_test = val_data[split_idx:]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f'data/val_backup_{timestamp}.json'
    try:
        shutil.copyfile(val_json, backup_path)
        print(f"Backed up validation set: {backup_path}")
    except Exception as e:
        print(f"Backup failed ({e}), continuing...")
    
    with open(output_val, 'w', encoding='utf-8') as f:
        json.dump(new_val, f, ensure_ascii=False, indent=2)
    with open(output_test, 'w', encoding='utf-8') as f:
        json.dump(new_test, f, ensure_ascii=False, indent=2)
    
    print(f"New validation set: {len(new_val)} samples → {output_val}")
    print(f"Test set: {len(new_test)} samples → {output_test}")
    
    print("\n" + "-"*70)
    print("Split completed")
    print("-"*70)
    print("Next steps:")
    print("  1. Train model using train_complete.py")
    print("  2. Evaluate on test set after training completes")
    print("\nImportant: Do not use test set during training!")
    print("="*70 + "\n")


def evaluate_test_set(model_path, 
                      model_version='v4',
                      test_json='data/test_new.json',
                      vocab_path='tibetan_vocab_optimized.txt',
                      batch_size=16):
    print("\n" + "="*70)
    print("Test Set Evaluation - Final Results")
    print("="*70)
    print(f"Model: {model_version.upper()}")
    print(f"Path: {model_path}")
    print(f"Test Set: {test_json}")
    print("="*70 + "\n")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not os.path.exists(test_json):
        print(f"Error: Test set not found: {test_json}")
        print("Run: python test.py --mode create-test")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    
    model_class = MODEL_DICT.get(model_version, CNNBiLSTMTransformer)
    model = model_class(vocab_size).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    test_transform = get_ocr_transforms(train=False)
    test_dataset = OCRDataset(test_json, vocab_path, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=2, 
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}\n")
    print("Evaluating...")
    
    all_preds = []
    all_true = []
    test_loss = 0
    test_batches = 0
    
    with torch.no_grad():
        for images, texts, text_lengths, raw_texts in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            
            logits = model(images)
            loss = compute_loss(logits, texts, text_lengths, blank=0)
            test_loss += loss.item()
            test_batches += 1
            
            preds = decode_predictions(logits, idx2char, blank=0)
            all_preds.extend(preds)
            all_true.extend(raw_texts)
    
    metrics = compute_all_metrics(all_preds, all_true)
    
    print("\n" + "="*70)
    print("Test Set Results")
    print("="*70)
    print(f"Test Loss: {test_loss/test_batches:.4f}\n")
    
    print("Core Metrics")
    print("-" * 70)
    print(f"1. CRR (Character Recognition Rate):      {metrics['CRR']:6.2f}%")
    print(f"   Correct/Total: {metrics['correct_chars']}/{metrics['total_chars']}")
    print()
    print(f"2. CER (Character Error Rate):            {metrics['CER']:6.2f}%")
    print(f"   Formula: (S + D + I) / N × 100%")
    print(f"   S={metrics['CER_substitutions']}, D={metrics['CER_deletions']}, I={metrics['CER_insertions']}")
    print()
    print(f"3. IER (Image Error Rate):                {metrics['IER']:6.2f}%")
    print(f"   Incorrect/Total: {metrics['incorrect_images']}/{metrics['total_images']}")
    print()
    print(f"4. DER (Diacritic Error Rate):            {metrics['DER']:6.2f}%")
    print(f"   Incorrect/Total: {metrics['incorrect_diacritics']}/{metrics['total_diacritics']}")
    print()
    print(f"5. SER (Sentence Error Rate):             {metrics['SER']:6.2f}%")
    print(f"   Incorrect/Total: {metrics['incorrect_sentences']}/{metrics['total_sentences']}")
    print("-" * 70)
    
    print("\nPrediction Samples (first 5):")
    for i in range(min(5, len(all_preds))):
        match = "✓" if all_preds[i] == all_true[i] else "✗"
        print(f"\n  {i+1}. {match}")
        print(f"      True: {all_true[i]}")
        print(f"      Pred: {all_preds[i]}")
    
    result_file = f'test_results_{model_version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Test Set Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Set: {test_json}\n")
        f.write(f"Test Loss: {test_loss/test_batches:.4f}\n\n")
        f.write("Core Metrics:\n")
        f.write(f"  CRR: {metrics['CRR']:.2f}%\n")
        f.write(f"  CER: {metrics['CER']:.2f}%\n")
        f.write(f"  IER: {metrics['IER']:.2f}%\n")
        f.write(f"  DER: {metrics['DER']:.2f}%\n")
        f.write(f"  SER: {metrics['SER']:.2f}%\n")
    
    print(f"\nResults saved to: {result_file}")
    
    print("\n" + "="*70)
    print("Evaluation completed!")
    print("="*70 + "\n")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tibetan OCR Test Set Tool')
    
    parser.add_argument('--mode', choices=['create-test', 'eval'], required=True,
                       help='Mode: create-test or eval')
    parser.add_argument('--model', '-m', choices=['v0', 'v1', 'v2', 'v3', 'v4'],
                       default='v4', help='Model version')
    parser.add_argument('--val-json', default='data/val_new.json',
                       help='Validation set path')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                       help='Test set ratio')
    parser.add_argument('--test-json', default='data/test_new.json',
                       help='Test set path')
    parser.add_argument('--model-path', help='Model path for eval mode')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--vocab', default='tibetan_vocab_optimized.txt',
                       help='Vocabulary path')
    
    args = parser.parse_args()
    
    if args.mode == 'create-test':
        print("\nThis will split validation set into validation and test sets")
        print("Test set should not be used during training")
        confirm = input("\nContinue? (yes/no): ")
        
        if confirm.lower() == 'yes':
            create_test_split(
                val_json=args.val_json,
                test_ratio=args.test_ratio,
                output_test=args.test_json
            )
        else:
            print("Cancelled")
    
    elif args.mode == 'eval':
        if not args.model_path:
            print("\nError: eval mode requires --model-path")
            print("\nExample:")
            print("  python test.py --mode eval \\")
            print("      --model-path saved_models/v4/best_model.pth \\")
            print("      --model v4")
            exit(1)
        
        evaluate_test_set(
            model_path=args.model_path,
            model_version=args.model,
            test_json=args.test_json,
            vocab_path=args.vocab,
            batch_size=args.batch_size
        )
