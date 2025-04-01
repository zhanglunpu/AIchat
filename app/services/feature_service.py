import torch
import torchvision.models as models
from pathlib import Path
import numpy as np
import cv2
from ..utils.image_utils import load_image, extract_features, normalize_vector
from .search_service import SearchService
from torchvision.models import ResNet50_Weights

class FeatureExtractor:
    def __init__(self):
        # 加载预训练的ResNet50模型
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 移除最后的全连接层
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # 如果有GPU则使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 初始化搜索服务
        base_dir = Path(__file__).parent.parent.parent
        static_dir = base_dir / 'static'
        self.search_service = SearchService(
            index_path=str(static_dir / 'search_index'),
            metadata_path=str(static_dir / 'herb_metadata.json')
        )

    async def extract_image_features(self, image_bytes):
        """从图像中提取特征向量"""
        # 加载和预处理图像
        image_tensor = load_image(image_bytes)
        image_tensor = image_tensor.to(self.device)
        
        # 提取特征
        features = extract_features(self.model, image_tensor)
        
        # 归一化特征向量
        normalized_features = normalize_vector(features)
        
        return normalized_features

    def __call__(self, image_bytes):
        """使类实例可以直接调用"""
        return self.extract_image_features(image_bytes)

    async def search_image(self, image_path, clip_weight=0.6, min_area_ratio=0.1):
        """
        搜索相似图像
        :param image_path: 图像路径
        :param clip_weight: CLIP 特征的权重 (0-1)
        :param min_area_ratio: 最小主要区域比例，低于此值的结果会被降权
        :return: (搜索结果列表, 热力图base64, 原图base64)
        """
        results, heatmap_base64, original_base64 = await self.search_service.search(
            image_path, 
            clip_weight=clip_weight, 
            min_area_ratio=min_area_ratio
        )
        return results, heatmap_base64, original_base64 