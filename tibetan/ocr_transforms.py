import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import os

class OCRTransforms:
    """专门为OCR任务设计的数据增强（无cv2依赖）"""
    
    def __init__(self, train=True, img_height=32, img_width=128):
        self.train = train
        self.img_height = img_height
        self.img_width = img_width
        
    def __getstate__(self):
        """确保类可被pickle序列化"""
        return self.__dict__
        
    def __setstate__(self, state):
        """确保类可被pickle反序列化"""
        self.__dict__.update(state)
        
    def geometric_transforms(self, image):
        """几何变换：模拟文档拍摄时的各种角度和变形"""
        if not self.train:
            return image
            
        # 轻微旋转 (-3到3度)
        if random.random() < 0.3:
            angle = random.uniform(-3, 3)
            image = TF.rotate(image, angle, fill=255)
        
        # 轻微缩放和剪切
        if random.random() < 0.2:
            shear = random.uniform(-5, 5)
            image = TF.affine(image, angle=0, translate=(0, 0), scale=1.0, shear=shear, fill=255)
        
        # 轻微平移
        if random.random() < 0.2:
            translate_x = random.randint(-2, 2)
            translate_y = random.randint(-1, 1)
            image = TF.affine(image, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0, fill=255)
        
        return image
    
    def quality_transforms(self, image):
        """图像质量变换：模拟不同的拍摄和扫描条件"""
        if not self.train:
            return image
        
        # 亮度调整
        if random.random() < 0.4:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # 对比度调整
        if random.random() < 0.4:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # 锐度调整
        if random.random() < 0.3:
            enhancer = ImageEnhance.Sharpness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # 轻微模糊 (模拟拍摄模糊)
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # 添加噪声
        if random.random() < 0.3:
            image = self._add_noise(image)
        
        return image
    
    def _add_noise(self, image):
        """添加噪声（PIL实现）"""
        img_array = np.array(image)
        
        # 高斯噪声
        if random.random() < 0.7:
            noise = np.random.normal(0, random.uniform(2, 8), img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        else:
            # 椒盐噪声
            noise_prob = random.uniform(0.001, 0.005)
            noisy_img = img_array.copy()
            
            # 盐噪声 (白点)
            salt_coords = np.random.random(img_array.shape) < noise_prob
            noisy_img[salt_coords] = 255
            
            # 椒噪声 (黑点)
            pepper_coords = np.random.random(img_array.shape) < noise_prob
            noisy_img[pepper_coords] = 0
        
        return Image.fromarray(noisy_img)
    
    def ocr_specific_transforms(self, image):
        """OCR特定的变换（简化版）"""
        if not self.train:
            return image
        
        # 背景纹理 (模拟纸张纹理)
        if random.random() < 0.1:
            image = self._add_paper_texture(image)
        
        # 简单的字符粗细变化（使用滤波器）
        if random.random() < 0.15:
            if random.random() < 0.5:
                # 让字符变粗
                image = image.filter(ImageFilter.MaxFilter(size=3))
            else:
                # 让字符变细
                image = image.filter(ImageFilter.MinFilter(size=3))
        
        return image
    
    def _add_paper_texture(self, image):
        """添加纸张纹理"""
        img_array = np.array(image)
        
        # 生成纸张纹理
        h, w = img_array.shape[:2]
        texture = np.random.normal(0, 3, (h, w))  # 减少纹理强度
        
        # 应用纹理
        if len(img_array.shape) == 3:
            for i in range(3):
                channel = img_array[:, :, i].astype(np.float32)
                channel += texture
                img_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.float32)
            img_array += texture
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def __call__(self, image):
        """应用所有变换"""
        # 确保输入是PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 应用变换序列
        if self.train:
            image = self.geometric_transforms(image)
            image = self.quality_transforms(image)
            image = self.ocr_specific_transforms(image)
        
        return image

def get_ocr_transforms(train=True, img_height=128, img_width=512):
    """获取OCR专用的数据变换pipeline"""
    
    if train:
        # 训练时的完整pipeline
        transform_list = [
            OCRTransforms(train=True, img_height=img_height, img_width=img_width),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    else:
        # 验证时的简单pipeline
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    
    return transforms.Compose(transform_list)

def get_light_augmentation():
    """轻量级数据增强"""
    def light_transform(image):
        ocr_transform = OCRTransforms(train=True)
        
        # 只应用部分变换
        if random.random() < 0.3:
            image = ocr_transform.quality_transforms(image)
        if random.random() < 0.2:
            image = ocr_transform.geometric_transforms(image)
            
        return image
    
    return transforms.Compose([
        light_transform,
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def get_heavy_augmentation():
    """重度数据增强"""
    return get_ocr_transforms(train=True)

def configure_dataloader_safely(dataset, batch_size=32, shuffle=True):
    """配置安全的数据加载器，解决多进程问题
    
    在Windows系统上使用PyTorch DataLoader时，多进程可能导致序列化错误
    此函数提供了安全的配置选项
    """
    import torch
    import platform
    
    # 检测操作系统
    is_windows = platform.system() == 'Windows'
    
    # Windows系统上默认禁用多进程
    # 其他系统使用默认设置
    num_workers = 0 if is_windows else 4
    
    # 其他安全选项
    persistent_workers = False if num_workers == 0 else True
    pin_memory = not is_windows
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory
    )

if __name__ == "__main__":
    # 创建简单的测试图像（含文字）
    # 创建白色背景图像
    try:
        # 尝试加载图像（使用标准路径格式）
        image_path = "data/images/0ec99b09-639.jpg"
        if not os.path.exists(image_path):
            # 尝试替代路径
            image_path = "data\\images\\0ec99b09-639.jpg"
            if not os.path.exists(image_path):
                print(f"警告：无法找到图像文件，创建测试图像")
                test_img = Image.new('RGB', (256, 64), color='white')
                draw = ImageDraw.Draw(test_img)
                # 使用默认字体
                font = ImageFont.load_default()
                draw.text((20, 20), "藏文OCR测试", fill="black", font=font)
            else:
                print(f"加载图像: {image_path}")
                test_img = Image.open(image_path)
        else:
            print(f"加载图像: {image_path}")
            test_img = Image.open(image_path)
    except Exception as e:
        print(f"加载图像时出错: {e}")
        test_img = Image.new('RGB', (256, 64), color='white')
        draw = ImageDraw.Draw(test_img)
    
    # 保存原始图像
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
    test_img.save("test_images/original.png")
    
    # 应用各种转换并保存
    transform = get_ocr_transforms(train=True)
    light_transform = get_light_augmentation()
    heavy_transform = get_heavy_augmentation()
    
    # 应用转换
    result = transform(test_img)
    
    # 将tensor转回PIL图像并保存
    result_img = TF.to_pil_image((result * 0.5 + 0.5).clamp(0, 1))
    result_img.save("test_images/transformed.png")
    
    # 应用轻度增强
    light_result = light_transform(test_img)
    light_result_img = TF.to_pil_image((light_result * 0.5 + 0.5).clamp(0, 1))
    light_result_img.save("test_images/light_augmented.png")
    
    # 应用重度增强
    heavy_result = heavy_transform(test_img)
    heavy_result_img = TF.to_pil_image((heavy_result * 0.5 + 0.5).clamp(0, 1))
    heavy_result_img.save("test_images/heavy_augmented.png")
    
    print(f"变换测试成功! 输出形状: {result.shape}")
    print(f"图像已保存至 'test_images' 文件夹") 