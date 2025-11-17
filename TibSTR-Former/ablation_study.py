import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import json
from typing import Dict, List
import glob

from model_v0 import CNNTransformerBase
from model_v1 import CNNTransformerResidual
from model_v2 import CNNTransformerFPN
from model_v3 import CNNTransformerAttention
from model_v4 import CNNBiLSTMTransformer
from data_loader import OCRDataset, collate_fn
from ocr_transforms import get_ocr_transforms

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

MODEL_DICT = {
    'v0': CNNTransformerBase,
    'v1': CNNTransformerResidual,
    'v2': CNNTransformerFPN,
    'v3': CNNTransformerAttention,
    'v4': CNNBiLSTMTransformer
}

MODEL_DESCRIPTIONS = {
    'v0': 'CNN + Transformer (Baseline)',
    'v1': '+ Residual Connections',
    'v2': '+ Feature Pyramid Network',
    'v3': '+ Attention Mechanism',
    'v4': '+ BiLSTM (Complete)'
}

TIBETAN_DIACRITICS = {
    'ི', 'ུ', 'ེ', 'ོ', 'ཱ', 'ྀ', 'ཻ', 'ཽ', 'ཾ', 'ྂ', 'ྃ',
    '྄', 'ཿ', '༹', '་', '༌', '།', '༎', '༏', '༐', '༑', 
    '༔', 'ཀ', 'ཁ', 'ག', 'ང', 'ཅ', 'ཆ', 'ཇ', 'ཉ', 'ཏ', 
    'ཐ', 'ད', 'ན', 'པ', 'ཕ', 'བ', 'མ', 'ཙ', 'ཚ', 'ཛ', 
    'ཝ', 'ཞ', 'ཟ', 'འ', 'ཡ', 'ར', 'ལ', 'ཤ', 'ས', 'ཧ', 
    'ཨ', '༡', '༢', '༣', '༤', '༥', '༦', '༧', '༨', '༩', 
    '༠', '༼', '༽', '།', '༄', '༅', '།', '།', '༈', '་', 
    '༜', '༄', 'ྱ', 'ྲ', 'ྀ', 'ཽ', 'ཻ', 'ྭ', 'ཾ', 'ླ', 
    'ཥ', 'ཋ', 'ཌ', 'ཊ', 'ཥ'
}

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

def compute_edit_distance_details(pred: str, true: str):
    m, n = len(true), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if true[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + 1
                )
    
    i, j = m, n
    insertions = deletions = substitutions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and true[i-1] == pred[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i-1][j] < dp[i][j-1]):
            deletions += 1
            i -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] < dp[i-1][j]):
            insertions += 1
            j -= 1
        else:
            substitutions += 1
            i -= 1
            j -= 1
    
    return dp[m][n], insertions, deletions, substitutions

def compute_all_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    total_samples = len(predictions)
    correct_count = 0
    total_chars = 0
    total_errors = 0
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    sequence_errors = 0
    
    insertion_errors = 0
    deletion_errors = 0
    diacritic_errors = 0
    total_diacritics = 0
    
    for pred, true in zip(predictions, ground_truths):
        if pred == true:
            correct_count += 1
        else:
            sequence_errors += 1
        
        true_len = len(true)
        total_chars += true_len
        
        edit_dist, ins, dels, subs = compute_edit_distance_details(pred, true)
        total_errors += edit_dist
        total_insertions += ins
        total_deletions += dels
        total_substitutions += subs
        
        if ins > 0:
            insertion_errors += 1
        if dels > 0:
            deletion_errors += 1
        
        for char in true:
            if char in TIBETAN_DIACRITICS:
                total_diacritics += 1
        
        for char in true:
            if char in TIBETAN_DIACRITICS and char not in pred:
                diacritic_errors += 1
    
    crr = (correct_count / total_samples * 100) if total_samples > 0 else 0
    cer = (total_errors / total_chars * 100) if total_chars > 0 else 0
    ier = (insertion_errors / total_samples * 100) if total_samples > 0 else 0
    deler = (deletion_errors / total_samples * 100) if total_samples > 0 else 0
    ser = (sequence_errors / total_samples * 100) if total_samples > 0 else 0
    der = (diacritic_errors / total_diacritics * 100) if total_diacritics > 0 else 0
    
    return {
        'CRR': crr,
        'CER': cer,
        'IER': ier,
        'DeletionER': deler,
        'SER': ser,
        'DER': der,
        'Total_Samples': total_samples,
        'Correct_Samples': correct_count,
        'Total_Chars': total_chars,
        'Total_Errors': total_errors,
        'Insertions': total_insertions,
        'Deletions': total_deletions,
        'Substitutions': total_substitutions,
        'Avg_Edit_Distance': total_errors / total_samples if total_samples > 0 else 0
    }

class AblationStudy:
    def __init__(self, vocab_path='tibetan_vocab_optimized.txt', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        self.idx2char = {idx: char for idx, char in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        print(f"Device: {self.device}")
        print(f"Vocab size: {self.vocab_size}")
    
    def evaluate_model(self, model_version: str, model_path: str, data_json: str, batch_size=16):
        print(f"\n{'='*70}")
        print(f"Evaluating {model_version}: {MODEL_DESCRIPTIONS[model_version]}")
        print(f"Model path: {model_path}")
        print(f"{'='*70}")
        
        torch.cuda.empty_cache()
        
        model_class = MODEL_DICT[model_version]
        model = model_class(self.vocab_size).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        transform = get_ocr_transforms(train=False)
        dataset = OCRDataset(data_json, 'tibetan_vocab_optimized.txt', transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
        
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for images, texts, text_lengths, raw_texts in loader:
                images = images.to(self.device)
                logits = model(images)
                preds = decode_predictions(logits, self.idx2char, blank=0)
                all_preds.extend(preds)
                all_trues.extend(raw_texts)
                
                del images, logits
                torch.cuda.empty_cache()
        
        metrics = compute_all_metrics(all_preds, all_trues)
        
        print(f"\nResults:")
        print(f"  CRR: {metrics['CRR']:.2f}%")
        print(f"  CER: {metrics['CER']:.2f}%")
        print(f"  IER: {metrics['IER']:.2f}%")
        print(f"  DER: {metrics['DER']:.2f}%")
        print(f"  SER: {metrics['SER']:.2f}%")
        print(f"  Avg Edit Distance: {metrics['Avg_Edit_Distance']:.2f}")
        
        del model
        torch.cuda.empty_cache()
        
        return metrics
    
    def run_ablation_study(self, model_paths: Dict[str, str], data_json: str, output_dir='ablation_results'):
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        for version in ['v0', 'v1', 'v2', 'v3', 'v4']:
            if version in model_paths:
                metrics = self.evaluate_model(version, model_paths[version], data_json)
                results[version] = metrics
        
        self.save_results(results, output_dir)
        self.generate_visualizations(results, output_dir)
        
        return results
    
    def save_results(self, results: Dict, output_dir: str):
        df_data = []
        for version in ['v0', 'v1', 'v2', 'v3', 'v4']:
            if version in results:
                metrics = results[version]
                df_data.append({
                    'Model': f"{version.upper()} ({MODEL_DESCRIPTIONS[version]})",
                    'CRR (%)': f"{metrics['CRR']:.2f}",
                    'CER (%)': f"{metrics['CER']:.2f}",
                    'IER (%)': f"{metrics['IER']:.2f}",
                    'DER (%)': f"{metrics['DER']:.2f}",
                    'SER (%)': f"{metrics['SER']:.2f}",
                    'Avg Edit Dist': f"{metrics['Avg_Edit_Distance']:.2f}"
                })
        
        df = pd.DataFrame(df_data)
        
        csv_path = os.path.join(output_dir, 'ablation_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {csv_path}")
        
        json_path = os.path.join(output_dir, 'ablation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {json_path}")
        
        print("\n" + "="*70)
        print("ABLATION STUDY RESULTS")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
    
    def generate_visualizations(self, results: Dict, output_dir: str):
        versions = []
        crr_values = []
        cer_values = []
        ier_values = []
        der_values = []
        ser_values = []
        
        for version in ['v0', 'v1', 'v2', 'v3', 'v4']:
            if version in results:
                versions.append(version.upper())
                metrics = results[version]
                crr_values.append(metrics['CRR'])
                cer_values.append(metrics['CER'])
                ier_values.append(metrics['IER'])
                der_values.append(metrics['DER'])
                ser_values.append(metrics['SER'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ablation Study: Model Comparison', fontsize=16, fontweight='bold')
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
        
        ax = axes[0, 0]
        bars = ax.bar(versions, crr_values, color=colors[:len(versions)], alpha=0.8, edgecolor='black')
        ax.set_ylabel('CRR (%)', fontsize=12, fontweight='bold')
        ax.set_title('Character Recognition Rate', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, crr_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax = axes[0, 1]
        bars = ax.bar(versions, cer_values, color=colors[:len(versions)], alpha=0.8, edgecolor='black')
        ax.set_ylabel('CER (%)', fontsize=12, fontweight='bold')
        ax.set_title('Character Error Rate', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, cer_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax = axes[0, 2]
        bars = ax.bar(versions, ier_values, color=colors[:len(versions)], alpha=0.8, edgecolor='black')
        ax.set_ylabel('IER (%)', fontsize=12, fontweight='bold')
        ax.set_title('Insertion Error Rate', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, ier_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax = axes[1, 0]
        bars = ax.bar(versions, der_values, color=colors[:len(versions)], alpha=0.8, edgecolor='black')
        ax.set_ylabel('DER (%)', fontsize=12, fontweight='bold')
        ax.set_title('Diacritic Error Rate', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, der_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax = axes[1, 1]
        bars = ax.bar(versions, ser_values, color=colors[:len(versions)], alpha=0.8, edgecolor='black')
        ax.set_ylabel('SER (%)', fontsize=12, fontweight='bold')
        ax.set_title('Sequence Error Rate', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, ser_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax = axes[1, 2]
        x = np.arange(len(versions))
        width = 0.15
        metrics_to_plot = [
            ('CRR', crr_values, '#2ecc71'),
            ('CER', cer_values, '#e74c3c'),
            ('IER', ier_values, '#f39c12'),
            ('DER', der_values, '#9b59b6'),
            ('SER', ser_values, '#3498db')
        ]
        for i, (metric_name, values, color) in enumerate(metrics_to_plot):
            offset = (i - 2) * width
            ax.bar(x + offset, values, width, label=metric_name, color=color, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(versions)
        ax.legend(loc='best', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'ablation_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {output_path}")

def find_best_model(model_dir: str) -> str:
    model_files = glob.glob(os.path.join(model_dir, 'best_model_*.pth'))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]

def main():
    parser = argparse.ArgumentParser(description='Ablation Study for Tibetan OCR Models')
    parser.add_argument('--data-json', default='data/test_new.json', help='Test dataset JSON path')
    parser.add_argument('--output-dir', default='ablation_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--v0-model', help='Path to v0 model')
    parser.add_argument('--v1-model', help='Path to v1 model')
    parser.add_argument('--v2-model', help='Path to v2 model')
    parser.add_argument('--v3-model', help='Path to v3 model')
    parser.add_argument('--v4-model', help='Path to v4 model')
    parser.add_argument('--auto-find', action='store_true', help='Auto find best models from saved_models/')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ABLATION STUDY - TIBETAN OCR MODELS")
    print("="*70)
    
    model_paths = {}
    
    if args.auto_find:
        print("\nAuto-finding best models...")
        for version in ['v0', 'v1', 'v2', 'v3', 'v4']:
            model_dir = f'saved_models/{version}'
            if os.path.exists(model_dir):
                try:
                    best_model = find_best_model(model_dir)
                    model_paths[version] = best_model
                    print(f"  {version}: {best_model}")
                except FileNotFoundError:
                    print(f"  {version}: Not found")
    else:
        if args.v0_model:
            model_paths['v0'] = args.v0_model
        if args.v1_model:
            model_paths['v1'] = args.v1_model
        if args.v2_model:
            model_paths['v2'] = args.v2_model
        if args.v3_model:
            model_paths['v3'] = args.v3_model
        if args.v4_model:
            model_paths['v4'] = args.v4_model
    
    if not model_paths:
        print("\n❌ Error: No model paths provided!")
        print("Use --auto-find or specify model paths manually:")
        print("  --v0-model, --v1-model, --v2-model, --v3-model, --v4-model")
        return
    
    study = AblationStudy(device=args.device)
    results = study.run_ablation_study(model_paths, args.data_json, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ ABLATION STUDY COMPLETE!")
    print("="*70)
    print(f"Results saved to: {args.output_dir}/")
    print("  - ablation_results.csv")
    print("  - ablation_results.json")
    print("  - ablation_comparison.png")

if __name__ == '__main__':
    main()
