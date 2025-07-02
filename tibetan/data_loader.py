import random
import sys

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
from line_profiler import LineProfiler

class OCRDataset(Dataset):
    def __init__(self, json_path, vocab_txt_path, transform=None, img_height=32, save_process_images=False, save_dir='processed_images'):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(vocab_txt_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(vocab)}

        self.samples = []
        self.img_height = img_height
        self.save_process_images = save_process_images
        self.save_dir = save_dir
        
        # 记录被过滤掉的样本数量
        filtered_count = 0
        total_count = 0
        
        # 如果需要保存处理图像，创建保存目录
        if self.save_process_images:
            os.makedirs(os.path.join(self.save_dir, 'original'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'processed'), exist_ok=True)

        for entry in data:
            total_count += 1
            # 支持新旧数据格式
            if 'image_path' in entry:
                # 新格式：直接使用image_path字段
                img_filename = os.path.basename(entry['image_path'])
                img_path = os.path.join('data/images', img_filename)
                if not os.path.exists(img_path):
                    # 尝试简化文件名：去掉image后和-之前的部分
                    if '-' in img_filename:
                        simplified_filename = img_filename.split('-', 1)[1]  # 取-后面的部分
                        simplified_path = os.path.join('data/images', simplified_filename)
                        if os.path.exists(simplified_path):
                            img_path = simplified_path
                        else:
                            print(f"简化文件名也不存在: {simplified_path}")
                            filtered_count += 1
                            continue

                
                # 如果路径已经包含data前缀，不要重复添加
                if not os.path.isabs(img_path) and not img_path.startswith('data/'):
                    img_path = os.path.join('data', img_path)
                
                # 处理新格式的文本和bbox
                if 'text' in entry and 'bbox' in entry:
                    text = entry['text']
                    bbox = entry['bbox']
                    
                    if not os.path.exists(img_path):
                        continue
                    
                    try:
                        image = Image.open(img_path).convert("RGB")
                        original_width, original_height = image.size
                        image_np = np.array(image)
                        
                        # 处理新格式的bbox（单个bbox对应单个文本）
                        x_min = int(bbox['x'] / 100 * original_width)
                        y_min = int(bbox['y'] / 100 * original_height)
                        width = int(bbox['width'] / 100 * original_width)
                        height = int(bbox['height'] / 100 * original_height)

                        x_max = min(x_min + width, original_width)
                        y_max = min(y_min + height, original_height)
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)

                        if x_max > x_min and y_max > y_min and (x_max - x_min) > 5 and (y_max - y_min) > 5:
                            crop_img = image_np[y_min:y_max, x_min:x_max]

                            h, w = crop_img.shape[:2]
                            pad_h = min(int(h * 0.1), 5)
                            pad_w = min(int(w * 0.1), 10)

                            padded_x_min = max(0, x_min - pad_w)
                            padded_y_min = max(0, y_min - pad_h)
                            padded_x_max = min(original_width, x_max + pad_w)
                            padded_y_max = min(original_height, y_max + pad_h)

                            padded_crop = image_np[padded_y_min:padded_y_max, padded_x_min:padded_x_max]

                            if padded_crop.size > 0 and len(text.strip()) > 0:
                                # 检查区域大小是否小于100*100
                                crop_height, crop_width = padded_crop.shape[:2]
                                if crop_width >= 80 and crop_height >= 80:
                                    self.samples.append({
                                        'image': padded_crop,
                                        'text': text.strip().replace(' ', '')
                                    })
                                else:
                                    filtered_count += 1
                    except Exception as e:
                        print(f"处理图像失败 {img_path}: {e}")
                        filtered_count += 1
                        continue
                    
        print(f"总样本数: {total_count}")
        print(f"过滤掉区域小于80*80的样本数: {filtered_count}")
        print(f"保留样本数: {len(self.samples)}")

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def visualize_samples(self, num_samples=5, save_path=None):
        """
        可视化随机样本
        :param num_samples: 要显示的样本数量
        :param save_path: 保存图像路径（可选）
        """
        if len(self.samples) == 0:
            print("没有可显示的样本")
            return

        # 随机选择样本
        indices = random.sample(range(len(self.samples)), min(num_samples, len(self.samples)))

        # 创建图像网格
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            sample = self.samples[idx]
            image = sample['image']
            text = sample['text']

            # 显示图像
            if image.ndim == 2:  # 灰度图
                axes[i].imshow(image, cmap='gray')
            else:  # RGB图
                axes[i].imshow(image)

            # 添加文本标题
            axes[i].set_title(f"'{text}'")
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"保存可视化结果到: {save_path}")
        else:
            plt.show()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        text = sample['text']

        # 转换为PIL图像进行变换
        image_pil = Image.fromarray(image)

        # 保存原始图像
        if self.save_process_images:
            image_path = os.path.join(self.save_dir, 'original', f'sample_{idx}.png')
            image_pil.save(image_path)

        # 保持宽高比调整大小
        w, h = image_pil.size
        ratio = w / float(h)
        new_h = self.img_height
        new_w = max(int(ratio * new_h), 32)  # 确保宽度至少为32

        if self.transform:
            image_pil = self.transform(image_pil)
        else:
            # 使用自适应宽度的变换
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((new_h, new_w)),  # 自适应宽度
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            image_pil = self.transform(image_pil)

        # 保存处理后的图像
        if self.save_process_images:
            # 转换张量为PIL图像并保存
            processed_img = (image_pil.squeeze().numpy() * 0.5 + 0.5) * 255
            processed_img = processed_img.astype(np.uint8)
            processed_pil = Image.fromarray(processed_img)
            processed_path = os.path.join(self.save_dir, 'processed', f'sample_{idx}.png')
            processed_pil.save(processed_path)

        # 文本转索引 - CTC训练不需要<sos>和<eos>标记
        text_idx = [self.char2idx.get(char, self.char2idx['<unk>']) for char in text]

        return image_pil, torch.tensor(text_idx), text


def collate_fn(batch):
    images, texts, raw_texts = zip(*batch)

    # 处理图像 - 允许动态宽度
    batch_size = len(images)
    channels = images[0].shape[0]
    height = images[0].shape[1]
    
    # 获取批次中的最大宽度
    widths = [img.shape[2] for img in images]
    max_width = max(widths)

    # 填充图像到统一宽度
    padded_images = torch.zeros(batch_size, channels, height, max_width)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.shape[2]] = img

    # 处理文本
    text_lengths = [len(t) for t in texts]
    max_text_len = max(text_lengths)
    padded_texts = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text

    return padded_images, padded_texts, text_lengths, raw_texts


def create_tibetan_char_map():
    """创建更完整的藏文字符映射表"""
    # 基本藏文字母
    tibetan_chars = "ཀཁགངཅཆཇཉཊཋཌཌྷཎཏཐདནཔཕབམཙཚཛཛྷཝཞཟའཡརལཤཥསཧཨ"

    # 藏文元音符号
    vowel_signs = "ིེོུ"

    # 藏文数字
    numbers = "༠༡༢༣༤༥༦༧༨༩"

    # 标点符号
    punctuation = "་།༄༅༆༇༈༉༊"

    # 组合字符
    all_chars = set(tibetan_chars)
    for char in tibetan_chars:
        for sign in vowel_signs:
            all_chars.add(char + sign)

    # 添加特殊字符
    char_list = sorted(list(all_chars)) + list(vowel_signs) + list(numbers) + list(punctuation)

    # 构建映射表
    char2idx = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3
    }

    # 添加藏文字符
    for idx, char in enumerate(char_list):
        char2idx[char] = idx + 4  # 从4开始，避免与特殊字符冲突

    return char2idx



def test_data_loader(json_path, vocab_txt_path, save_process_images=False):
    with open(vocab_txt_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}

    # 创建数据集
    try:
        dataset = OCRDataset(
            json_path, 
            vocab_txt_path, 
            transform=None,
            save_process_images=save_process_images,
            save_dir='images/process_visualization'
        )
        

        if len(dataset) == 0:
            print("错误: 数据集为空，请检查数据路径和格式")
            return

        # 可视化样本
        print("\n随机样本可视化:")
        dataset.visualize_samples(num_samples=min(5, len(dataset)), save_path="images/data_samples.png")

        # 测试数据加载器
        print("\n测试数据加载器批次处理:")
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn
        )

        # 获取一个批次
        for batch_idx, (images, texts, text_lengths, raw_texts) in enumerate(loader):
            if images.numel() == 0:  # 跳过空批次
                continue

            print(f"\n批次 {batch_idx + 1}:")
            print(f"图像张量形状: {images.shape}")
            print(f"文本张量形状: {texts.shape}")
            print(f"文本长度: {text_lengths}")
            print(f"原始文本: {raw_texts}")

            # 显示每个样本的解码结果
            for sample_idx in range(len(raw_texts)):
                print(f"\n样本 {sample_idx + 1}:")
                print(f"原始文本: {raw_texts[sample_idx]}")
                print(f"文本索引: {texts[sample_idx].tolist()}")

                # 文本索引解码
                decoded_text = []
                for idx in texts[sample_idx]:
                    if idx.item() == 0:  # 跳过填充
                        continue
                    char = idx2char.get(idx.item(), '�')  # 使用�表示未知字符
                    decoded_text.append(char)

                # 移除特殊标记
                decoded_text_str = ''.join(decoded_text)
                decoded_text_str = decoded_text_str.replace('<sos>', '').replace('<eos>', '')

                print(f"解码文本: {decoded_text_str}")
                print(f"解码是否正确: {decoded_text_str == raw_texts[sample_idx]}")
            if not decoded_text_str == raw_texts[sample_idx]:
                break


    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print(f"数据集大小: {len(dataset)} 个样本")

def update_tibetan_vocab_with_dataset(json_paths, vocab_txt_path, save_path):
    """
    统计所有json文件中的藏文字符，如果词表中没有则补充，并保存为新txt。
    :param json_paths: 列表，所有json标注文件路径
    :param vocab_txt_path: 现有词表文件路径
    :param save_path: 要保存的词表新路径
    """
    # 1. 读入现有词表
    with open(vocab_txt_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    vocab_set = set(vocab)
    specials = {'<blank>', '<sos>', '<eos>', '<unk>'}

    # 2. 扫描所有json，统计所有出现的字符
    all_chars = set()
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                # 支持新旧数据格式
                if 'text' in entry:
                    # 新格式：直接使用text字段
                    text = entry['text']
                    all_chars.update(list(text.strip()))
                elif 'transcription' in entry:
                    # 旧格式：使用transcription字段
                    texts = entry['transcription']
                    if isinstance(texts, list):
                        for t in texts:
                            all_chars.update(list(t.strip()))
                    else:
                        all_chars.update(list(texts.strip()))

    # 3. 找出未在vocab中的字符
    missing_chars = sorted(all_chars - vocab_set - specials)
    print(f"【补充到词表的字符共 {len(missing_chars)} 个】:", ''.join(missing_chars))

    # 4. 构造最终词表
    #   先保留特殊符号头部（如有），再保留原有字符，最后补充新字符
    final_vocab = []
    # 保证特殊符号在前面
    for token in ['<pad>', '<sos>', '<eos>', '<unk>']:
        if token not in vocab:
            final_vocab.append(token)
    for token in vocab:
        if token not in final_vocab:
            final_vocab.append(token)
    for char in missing_chars:
        final_vocab.append(char)

    # 5. 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        for char in final_vocab:
            f.write(char + '\n')
    print(f"已保存补全后的词表，共{len(final_vocab)}项 -> {save_path}")




def save_tibetan_char_map_to_txt(save_path):
    """
    用 create_tibetan_char_map 函数生成字符集并保存为txt文件（每行一个字符）。
    """
    # 你的原始词表生成代码
    tibetan_chars = "ཀཁགངཅཆཇཉཊཋཌཌྷཎཏཐདནཔཕབམཙཚཛཛྷཝཞཟའཡརལཤཥསཧཨ"
    vowel_signs = "ིེོུ"
    numbers = "༠༡༢༣༤༥༦༧༨༩"
    punctuation = "་།༄༅༆༇༈༉༊"

    all_chars = set(tibetan_chars)
    for char in tibetan_chars:
        for sign in vowel_signs:
            all_chars.add(char + sign)

    char_list = sorted(list(all_chars)) + list(vowel_signs) + list(numbers) + list(punctuation)

    # 你要的特殊符号（推荐加在前面，OCR工程约定）
    specials = ['<pad>', '<sos>', '<eos>', '<unk>']

    with open(save_path, 'w', encoding='utf-8') as f:
        for token in specials:
            f.write(token + '\n')
        for char in char_list:
            f.write(char + '\n')
    print(f"词表已保存到 {save_path}，共 {len(char_list) + len(specials)} 行")

# 用法




if __name__ == '__main__':
    # 测试数据加载器
    print("开始测试数据加载器...")
    save_tibetan_char_map_to_txt('tibetan_vocab.txt')
    
    update_tibetan_vocab_with_dataset(
        json_paths=['data/train_new.json', 'data/val_new.json'],
        vocab_txt_path='tibetan_vocab.txt',
        save_path='tibetan_vocab_full.txt'
    )
    
    # 启用图像处理可视化功能
    print("测试数据加载并保存处理前后的图像...")
    test_data_loader('data/train_new.json', 'tibetan_vocab_full.txt', save_process_images=True)
    print(f"处理前后的图像已保存到 'images/process_visualization' 文件夹")