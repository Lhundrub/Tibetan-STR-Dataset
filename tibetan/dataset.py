import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from PIL import Image
import json
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class TibetanDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = entry['transcription']
        print(type(text))
        img_filename = os.path.basename(entry['ocr'])
        img_path = os.path.join('data/images', img_filename)
        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size
        
        # 转换bbox为Albumentations格式
        bboxes = []
        labels = []  # 添加单独的标签列表
        for bbox in entry['bbox']:
            x_min = bbox['x']/100 * original_width
            y_min = bbox['y']/100 * original_height
            width = bbox['width']/100 * original_width
            height = bbox['height']/100 * original_height
            
            # 确保坐标不超出图像边界
            x_max = min(x_min + width, original_width)
            y_max = min(y_min + height, original_height)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            
            # 确保边界框有效（宽度和高度大于0）
            if x_max > x_min and y_max > y_min:
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  # 添加到单独的标签列表
        
        # 如果没有有效的边界框，创建一个虚拟的小边界框，稍后会过滤掉
        if len(bboxes) == 0:
            bboxes.append([0, 0, 1, 1])
            labels.append(1)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(
                image=np.array(image),
                bboxes=bboxes,
                labels=labels  # 传递标签列表
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']  # 获取转换后的标签
        
        # 过滤无效的bbox并转换为tensor
        boxes = []
        valid_labels = []  # 用于存储有效bbox的标签
        for i, box in enumerate(bboxes):
            x_min, y_min, x_max, y_max = box
            # 再次检查变换后的边界框有效性
            if x_max > x_min and y_max > y_min:  # 有效性检查
                boxes.append([x_min, y_min, x_max, y_max])
                valid_labels.append(labels[i])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            valid_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            valid_labels = torch.as_tensor(valid_labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": valid_labels,  # 使用有效标签
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "texts": text
        }
        return image, target
    

class OCRDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        for entry in data:
            img_filename = os.path.basename(entry['ocr'])
            img_path = os.path.join('data/images', img_filename)
            
            # 确保图像文件存在
            if not os.path.exists(img_path):
                continue
                
            image = Image.open(img_path).convert("RGB")
            original_width, original_height = image.size
            image_np = np.array(image)
            
            # 处理每个边界框和对应的文本
            for i, bbox in enumerate(entry['bbox']):
                x_min = int(bbox['x']/100 * original_width)
                y_min = int(bbox['y']/100 * original_height)
                width = int(bbox['width']/100 * original_width)
                height = int(bbox['height']/100 * original_height)
                
                # 确保坐标不超出图像边界
                x_max = min(x_min + width, original_width)
                y_max = min(y_min + height, original_height)
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                
                # 确保边界框有效（宽度和高度大于0且足够大）
                if x_max > x_min and y_max > y_min and (x_max - x_min) > 5 and (y_max - y_min) > 5:
                    # 裁剪边界框部分的图像
                    crop_img = image_np[y_min:y_max, x_min:x_max]
                    
                    # 处理文本
                    if isinstance(entry['transcription'], list):
                        if i < len(entry['transcription']):
                            text = entry['transcription'][i]
                        else:
                            continue  # 跳过没有对应文本的边界框
                    else:
                        text = entry['transcription']
                    
                    # 添加边距以改善识别效果
                    h, w = crop_img.shape[:2]
                    # 计算边距，确保不超过原始尺寸的10%
                    pad_h = min(int(h * 0.1), 5)
                    pad_w = min(int(w * 0.1), 10)
                    
                    # 检查是否可以扩展边界框
                    padded_x_min = max(0, x_min - pad_w)
                    padded_y_min = max(0, y_min - pad_h)
                    padded_x_max = min(original_width, x_max + pad_w)
                    padded_y_max = min(original_height, y_max + pad_h)
                    
                    # 使用扩展的边界框
                    padded_crop = image_np[padded_y_min:padded_y_max, padded_x_min:padded_x_max]
                    
                    # 添加样本，包含裁剪的图像和对应的文本
                    if padded_crop.size > 0 and len(text.strip()) > 0:  # 确保裁剪的图像和文本非空
                        self.samples.append({
                            'image': padded_crop,
                            'text': text.strip()
                        })
        
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        text = sample['text']
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, text

def get_transform(train=True):
    transform_list = [
        A.Resize(800, 800),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    if train:
        transform_list.insert(1, A.HorizontalFlip(p=0.5))
        # 添加更多数据增强可能导致边界框失效，所以添加有限的增强
        transform_list.insert(1, A.RandomBrightnessContrast(p=0.2))
    
    return A.Compose(
        transform_list,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0.0,  # 减小最小面积限制
            min_visibility=0.0,  # 减小最小可见度限制
            label_fields=['labels']  # 确保这里使用的标签字段名与传入数据相匹配
        )
    )

def get_ocr_transform(train=True):
    """
    获取OCR数据集的变换
    """
    transform_list = [
        A.Resize(height=128, width=512),  # OCR通常使用宽高比更适合文本的尺寸
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    if train:
        transform_list.insert(1, A.RandomBrightnessContrast(p=0.2))
        transform_list.insert(1, A.GaussNoise(var_limit=(10.0, 50.0), p=0.1))
    
    return A.Compose(transform_list)

if __name__ == "__main__":
    # 使用示例
    dataset = TibetanDataset(
        'data/result.json',
        transform=get_transform(train=True)
    )
    ocr_dataset = OCRDataset(
        'data/result.json',
        transform=get_ocr_transform(train=True)
    )
    print("TibetanDataset total samples:", len(dataset))
    print("OCRDataset total samples:", len(ocr_dataset))
    
    # 查看OCRDataset样本
    def visualize_ocr_samples(dataset, num_samples=5):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        for i in range(min(num_samples, len(dataset))):
            image, text = dataset[i]
            # 如果是tensor，转换回numpy
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
                # 反标准化
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                # 确保像素值在[0,1]范围内
                image = np.clip(image, 0, 1)
            
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(image)
            plt.title(f"Text: {text}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_images/ocr_samples.png')
        print(f"保存OCR样本可视化到 test_images/ocr_samples.png")
    
    # 查看数据增强效果
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    def visualize(idx):
        image, target = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 反标准化
        
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        for box in target['boxes']:
            x_min, y_min, x_max, y_max = box.tolist()
            rect = patches.Rectangle(
                (x_min, y_min), x_max-x_min, y_max-y_min,
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.savefig('test_images/1.png')

    visualize(0)
    visualize_ocr_samples(ocr_dataset)