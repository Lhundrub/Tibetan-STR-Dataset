import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from dataset import TibetanDataset, get_transform
from detect import get_faster_rcnn_model

def parse_args():
    parser = argparse.ArgumentParser(description='藏文目标检测模型评估')
    parser.add_argument('--model_path', type=str, default='saved_models/best_detection_model.pth',
                        help='模型权重文件路径')
    parser.add_argument('--data_path', type=str, default='data/val.json',
                        help='测试数据集路径')
    parser.add_argument('--output_path', type=str, default='output/evaluation_results.json',
                        help='评估结果输出路径')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IOU阈值，判断检测为正确的最小IOU值')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器的工作线程数')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='置信度阈值，只考虑高于此阈值的预测')
    
    return parser.parse_args()

def calculate_ap(precision, recall):
    """
    计算平均精度 (AP)，使用11点插值法
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0
    return ap

def evaluate(model, data_loader, device, iou_threshold=0.5, confidence_threshold=0.5):
    """
    评估模型性能，计算精度、召回率和mAP
    """
    model.eval()
    
    # 存储所有检测结果
    total_map = 0.0
    total_samples = 0
    
    # 存储各个阈值下的召回率和精度
    thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, ..., 0.95]
    aps_per_threshold = {t: [] for t in thresholds}
    
    class_map = 0.0
    class_samples = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='评估模型'):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 获取预测结果
            outputs = model(images)
            
            # 计算每个样本的AP
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_labels = output['labels']
                
                gt_boxes = targets[i]['boxes']
                gt_labels = targets[i]['labels']
                
                # 如果没有真实框或预测框，跳过
                if len(gt_boxes) == 0:
                    continue
                
                # 只考虑得分高于阈值的预测框
                high_score_indices = pred_scores >= confidence_threshold
                pred_boxes = pred_boxes[high_score_indices]
                pred_scores = pred_scores[high_score_indices]
                pred_labels = pred_labels[high_score_indices]
                
                if len(pred_boxes) == 0:
                    # 没有预测框，AP=0
                    total_map += 0
                    total_samples += 1
                    class_samples += 1
                    for t in thresholds:
                        aps_per_threshold[t].append(0)
                    continue
                
                # 排序预测框，使得得分较高的框在前面
                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_indices]
                pred_scores = pred_scores[sorted_indices]
                pred_labels = pred_labels[sorted_indices]
                
                # 计算不同IoU阈值下的AP
                for threshold in thresholds:
                    # 计算IoU
                    ious = box_iou(pred_boxes, gt_boxes)
                    
                    # 初始化TP和FP
                    tp = torch.zeros(len(pred_boxes))
                    fp = torch.zeros(len(pred_boxes))
                    
                    # 匹配预测框和真实框
                    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
                    
                    for j in range(len(pred_boxes)):
                        # 找到与当前预测框具有最大IoU的真实框
                        max_iou, max_idx = torch.max(ious[j], dim=0)
                        
                        if max_iou >= threshold and not gt_matched[max_idx]:
                            tp[j] = 1
                            gt_matched[max_idx] = True
                        else:
                            fp[j] = 1
                    
                    # 计算累积TP和FP
                    tp_cumsum = torch.cumsum(tp, dim=0)
                    fp_cumsum = torch.cumsum(fp, dim=0)
                    
                    # 计算精度和召回率
                    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                    recall = tp_cumsum / len(gt_boxes)
                    
                    # 添加精度为1.0，召回率为0.0的起始点
                    precision = torch.cat([torch.tensor([1.0]), precision])
                    recall = torch.cat([torch.tensor([0.0]), recall])
                    
                    # 计算AP
                    ap = calculate_ap(precision.numpy(), recall.numpy())
                    aps_per_threshold[threshold].append(ap)
                    
                    # 只记录IoU=0.5时的AP
                    if threshold == 0.5:
                        total_map += ap
                        total_samples += 1
                        
                        # 记录类别为1的AP (藏文文本)
                        if 1 in gt_labels:
                            class_map += ap
                            class_samples += 1
    
    # 计算平均结果
    mean_ap = total_map / max(1, total_samples)
    class_mean_ap = class_map / max(1, class_samples)
    
    # 计算各个IoU阈值下的mAP
    map_per_threshold = {t: np.mean(aps) for t, aps in aps_per_threshold.items() if aps}
    
    # 计算mAP@[0.5:0.95] (COCO指标)
    map_50_95 = np.mean(list(map_per_threshold.values()))
    
    return {
        'mAP@0.5': mean_ap,
        'class_AP@0.5': class_mean_ap,
        'mAP@[0.5:0.95]': map_50_95,
        'mAP_per_threshold': map_per_threshold
    }

def plot_precision_recall_curve(precision, recall, save_path='output/precision_recall_curve.png'):
    """绘制精度-召回率曲线"""
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig(save_path)
    plt.close()

def visualize_results(results, save_path='output/evaluation_plot.png'):
    """可视化评估结果"""
    # 提取不同IoU阈值的mAP值
    thresholds = list(results['mAP_per_threshold'].keys())
    maps = list(results['mAP_per_threshold'].values())
    
    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, maps, 'ro-', linewidth=2)
    plt.title('mAP at Different IoU Thresholds')
    plt.xlabel('IoU Threshold')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.xticks(thresholds)
    plt.savefig(save_path)
    plt.close()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据集
    dataset = TibetanDataset(args.data_path, get_transform(train=False))
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)), num_workers=args.num_workers
    )
    
    # 加载模型
    model = get_faster_rcnn_model(num_classes=2)  # 背景 + 藏文文本
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # 评估模型
    print(f"开始评估模型 {args.model_path} 在数据集 {args.data_path} 上的性能...")
    results = evaluate(model, data_loader, device, args.iou_threshold, args.confidence_threshold)
    
    # 打印结果
    print(f"评估完成！")
    print(f"藏文类别的AP@0.5: {results['class_AP@0.5']:.4f}")
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"mAP@[0.5:0.95]: {results['mAP@[0.5:0.95]']:.4f}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 保存结果
    with open(args.output_path, 'w') as f:
        # 将浮点数转换为常规Python数字，以便JSON序列化
        results_json = {
            'mAP@0.5': float(results['mAP@0.5']),
            'class_AP@0.5': float(results['class_AP@0.5']),
            'mAP@[0.5:0.95]': float(results['mAP@[0.5:0.95]']),
            'mAP_per_threshold': {float(k): float(v) for k, v in results['mAP_per_threshold'].items()}
        }
        json.dump(results_json, f, indent=4)
    
    # 可视化结果
    visualize_results(results, save_path='output/evaluation_plot.png')
    
    print(f"评估结果已保存至 {args.output_path}")
    print(f"评估可视化结果已保存至 output/evaluation_plot.png")

if __name__ == "__main__":
    main() 