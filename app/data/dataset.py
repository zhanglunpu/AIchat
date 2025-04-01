import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import random

class ChineseMedicineDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, test_size=0.2, random_state=42):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 加载类别信息
        self.classes = sorted([d.name for d in self.data_dir.iterdir() 
                             if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 收集所有图片路径
        all_samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.[jp][pn][g]'):  # 支持jpg,jpeg,png
                all_samples.append((str(img_path), self.class_to_idx[class_name]))
        
        # 划分训练集和测试集
        train_samples, test_samples = train_test_split(
            all_samples, 
            test_size=test_size,
            random_state=random_state,
            stratify=[label for _, label in all_samples]
        )
        
        self.samples = train_samples if split == 'train' else test_samples
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 返回数据集中的另一个随机样本
            return self[random.randint(0, len(self)-1)]
        
def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """创建训练和验证数据加载器"""
    train_dataset = ChineseMedicineDataset(data_dir, split='train')
    test_dataset = ChineseMedicineDataset(data_dir, split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.classes 