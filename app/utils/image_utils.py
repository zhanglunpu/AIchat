import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from io import BytesIO

# 图像预处理转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_bytes):
    """从字节数据加载图像并进行预处理"""
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)

def extract_features(model, image_tensor):
    """使用模型提取图像特征"""
    with torch.no_grad():
        features = model(image_tensor)
        # 获取倒数第二层的特征作为图像表示
        if isinstance(features, torch.Tensor):
            features = features.squeeze()
        else:
            features = features[-2].squeeze()
    return features.numpy()

def normalize_vector(vector):
    """L2归一化向量"""
    return vector / np.linalg.norm(vector) 