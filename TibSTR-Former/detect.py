import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from dataset import TibetanDataset, get_transform
import matplotlib.patches as patches
import random

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 创建FasterRCNN模型
def get_faster_rcnn_model(num_classes):
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # 获取分类器输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 替换预测头以适应我们的类别数量（背景 + 藏文文本）
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# 计算平均精度
def calculate_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """计算mAP (mean Average Precision)"""
    if len(gt_boxes) == 0:
        return 0.0
    
    # 如果没有预测框，则精度为0
    if len(pred_boxes) == 0:
        return 0.0
    
    # 排序预测框，使得得分较高的框在前面
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    
    # 计算IoU
    ious = torch.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            # 计算IoU
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            x2 = min(pred_box[2], gt_box[2])
            y2 = min(pred_box[3], gt_box[3])
            
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            intersection = width * height
            
            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            union = pred_area + gt_area - intersection
            
            ious[i, j] = intersection / union
    
    # 初始化TP和FP
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    
    # 匹配预测框和真实框
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    
    for i in range(len(pred_boxes)):
        # 找到与当前预测框具有最大IoU的真实框
        max_iou, max_idx = torch.max(ious[i], dim=0)
        
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp[i] = 1
            gt_matched[max_idx] = True
        else:
            fp[i] = 1
    
    # 计算累积TP和FP
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    # 计算精度和召回率
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_boxes)
    
    # 添加精度为1.0，召回率为0.0的起始点
    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])
    
    # 计算AP (使用11点插值法)
    ap = 0.0
    for t in torch.arange(0.0, 1.1, 0.1):
        if torch.sum(recall >= t) == 0:
            p = 0
        else:
            p = torch.max(precision[recall >= t])
        ap = ap + p / 11.0
    
    return ap.item()

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 计算损失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # 更新模型参数
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def validate(model, data_loader, device):
    model.eval()
    total_map = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 获取预测结果
            outputs = model(images)
            
            # 计算每个样本的mAP
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_labels = output['labels']
                
                gt_boxes = targets[i]['boxes']
                gt_labels = targets[i]['labels']
                
                # 只考虑得分高于阈值的预测框
                score_threshold = 0.5
                high_score_indices = pred_scores >= score_threshold
                pred_boxes = pred_boxes[high_score_indices]
                pred_scores = pred_scores[high_score_indices]
                pred_labels = pred_labels[high_score_indices]
                
                ap = calculate_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
                total_map += ap
                num_samples += 1
    
    return total_map / max(1, num_samples)

def plot_metrics(train_losses, val_maps, save_path='metrics.png'):
    """绘制训练损失和验证mAP曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # 绘制验证mAP
    plt.subplot(1, 2, 2)
    plt.plot(val_maps, 'r-', label='Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(model, dataset, device, num_samples=5, output_dir='output/visualizations'):
    """可视化模型预测结果与真实标签的对比"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择样本
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 获取样本
            image, target = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            # 获取预测
            prediction = model(image_tensor)[0]
            
            # 转换为numpy进行可视化
            image_np = image.permute(1, 2, 0).cpu().numpy()
            # 标准化图像用于显示
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # 左侧：显示真实标签
            ax1.imshow(image_np)
            ax1.set_title('ground truth')
            
            # 绘制真实框
            for box in target['boxes'].cpu().numpy():
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
                ax1.add_patch(rect)
            
            # 右侧：显示预测标签
            ax2.imshow(image_np)
            ax2.set_title('prediction')
            
            # 设置得分阈值
            score_threshold = 0.5
            
            # 绘制预测框
            for box, score in zip(prediction['boxes'].cpu().numpy(), prediction['scores'].cpu().numpy()):
                if score >= score_threshold:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    ax2.add_patch(rect)
                    # 显示得分
                    ax2.text(x1, y1, f'{score:.2f}', bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/epoch_sample_{i+1}.png')
            plt.close(fig)
    
    print(f'已保存{num_samples}个样本的可视化结果到{output_dir}目录')

def main():
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 定义超参数
    num_classes = 2  # 背景 + 藏文文本
    batch_size = 8
    num_epochs = 20
    learning_rate = 0.005
    weight_decay = 0.0005
    
    # 加载数据集
    train_dataset = TibetanDataset(
        'data/train.json',
        transform=get_transform(train=True)
    )
    
    val_dataset = TibetanDataset(
        'data/val.json',
        transform=get_transform(train=False)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  
        collate_fn=lambda x: tuple(zip(*x))  # 自定义collate函数
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))  # 自定义collate函数
    )
    
    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    # 创建模型
    model = get_faster_rcnn_model(num_classes)
    model.to(device)
    
    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    # 用于记录训练和验证指标
    train_losses = []
    val_maps = []
    best_map = 0.0
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_map = validate(model, val_loader, device)
        val_maps.append(val_map)
        
        print(f'训练损失: {train_loss:.4f}, 验证mAP: {val_map:.4f}')
        
        # 可视化预测结果
        if epoch % 5 == 0:
            visualize_predictions(
                model, 
                val_dataset, 
                device, 
                num_samples=5, 
                output_dir=f'output/visualizations/epoch_{epoch+1}'
            )
        
        # 更新学习率
        lr_scheduler.step()
        
        # 保存最佳模型
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), 'saved_models/best_detection_model.pth')
            print(f'模型已保存: saved_models/best_detection_model.pth (mAP: {best_map:.4f})')
        
        # 每个epoch保存一次模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_map': val_map,
        }, f'saved_models/detection_checkpoint_epoch_{epoch+1}.pth')
        
        # 保存指标
        with open('output/metrics.json', 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_maps': val_maps,
            }, f)
        
        # 绘制指标曲线
        plot_metrics(train_losses, val_maps, save_path='output/metrics.png')

if __name__ == '__main__':
    main()
