from towhee import AutoPipes
import numpy as np
import faiss
import json
import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import base64
import cv2
from io import BytesIO

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradient = None
        
        # 注册钩子
        def save_gradient(module, grad_input, grad_output):
            self.gradient = grad_output[0]
            
        def forward_hook(module, input, output):
            self.feature_maps = output
            
        # 获取最后一个卷积层
        target_layer = list(self.model.children())[-4]  # ResNet50 的最后一个卷积块
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(save_gradient)
    
    def get_cam(self, input_tensor, original_image):
        """
        生成类激活图
        :param input_tensor: 输入图像张量
        :param original_image: 原始PIL图像
        :return: (热力图base64, 原图base64, 区域比例)
        """
        # 清除之前的梯度和缓存
        self.feature_maps = None
        self.gradient = None
        self.model.zero_grad()
        
        # 前向传播
        with torch.set_grad_enabled(True):
            output = self.model(input_tensor)
            score = output.max()  # 获取最高分数
            score.backward()  # 反向传播
        
        # 检查是否成功获取梯度和特征图
        if self.gradient is None or self.feature_maps is None:
            print("Warning: Failed to get gradients or feature maps")
            return None, None, 0.1
        
        # 计算权重
        weights = torch.mean(self.gradient, dim=(2, 3))  # GAP
        
        # 生成CAM
        batch_size, n_channels, height, width = self.feature_maps.shape
        cam = torch.zeros((batch_size, height, width), dtype=torch.float32, device=self.feature_maps.device)
        
        for i in range(n_channels):
            cam += weights[:, i].view(-1, 1, 1) * self.feature_maps[:, i, :, :]
        
        # ReLU
        cam = torch.maximum(cam, torch.zeros_like(cam))
        
        # 归一化
        if cam.max() - cam.min() > 1e-7:  # 避免除以0
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # 计算区域比例
        threshold = 0.5  # 激活阈值
        active_pixels = torch.sum(cam > threshold).item()
        total_pixels = cam.numel()
        area_ratio = active_pixels / total_pixels
        
        # 确保区域比例不会太小
        area_ratio = max(0.1, area_ratio)
        
        # 转换为numpy数组并调整大小
        cam = cam.detach().cpu().numpy()[0]  # 只取第一个batch
        cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
        cam = (cam * 255).astype(np.uint8)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        
        # 将原图转换为numpy数组
        original_array = np.array(original_image)
        original_array = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
        
        # 叠加热力图
        overlay = cv2.addWeighted(original_array, 0.7, heatmap, 0.3, 0)
        
        # 将原图和热力图转换为base64
        _, original_buffer = cv2.imencode('.jpg', original_array)
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        _, heatmap_buffer = cv2.imencode('.jpg', overlay)
        heatmap_base64 = base64.b64encode(heatmap_buffer).decode('utf-8')
        
        return heatmap_base64, original_base64, area_ratio

class SearchService:
    def __init__(self, index_path, metadata_path):
        """
        初始化搜索服务
        :param index_path: FAISS 索引文件路径
        :param metadata_path: 元数据文件路径
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # 初始化 CLIP 模型
        self.clip_pipeline = AutoPipes.pipeline('image-embedding')
        
        # 初始化 ResNet50 模型
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化 Grad-CAM
        self.grad_cam = GradCAM(self.resnet)
        
        # 加载索引和元数据
        if (os.path.exists(index_path + '.clip') and 
            os.path.exists(index_path + '.resnet') and 
            os.path.exists(metadata_path)):
            self.load_index()
        else:
            self.clip_index = None
            self.resnet_index = None
            self.metadata = []

    async def extract_features(self, image_path):
        """
        提取图像特征
        :param image_path: 图像路径
        :return: (clip_features, resnet_features, area_ratio, heatmap_base64, original_base64)
        """
        # 提取 CLIP 特征
        clip_features = self.clip_pipeline(image_path).get()[0]
        
        # 提取 ResNet 特征和计算 Grad-CAM
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.inference_mode():
            resnet_features = self.resnet(image_tensor)
        
        # 计算 Grad-CAM 和区域比例
        heatmap_base64, original_base64, area_ratio = self.grad_cam.get_cam(image_tensor, image)
        
        resnet_features = resnet_features.squeeze().cpu().numpy()
        
        return clip_features, resnet_features, area_ratio, heatmap_base64, original_base64

    async def add_vectors(self, image_paths, metadata_list, rebuild=False):
        """
        添加向量到索引
        :param image_paths: 图像路径列表
        :param metadata_list: 元数据列表
        :param rebuild: 是否重建索引(清空原有内容)
        """
        clip_features_list = []
        resnet_features_list = []
        
        # 检查重复数据
        existing_ids = set()
        if not rebuild and self.metadata:
            existing_ids = {item['id'] for item in self.metadata}
        
        # 过滤重复的数据
        filtered_paths = []
        filtered_metadata = []
        for path, meta in zip(image_paths, metadata_list):
            if rebuild or meta['id'] not in existing_ids:
                filtered_paths.append(path)
                filtered_metadata.append(meta)
                if not rebuild:
                    existing_ids.add(meta['id'])
        
        # 如果没有新数据要添加，直接返回
        if not filtered_paths:
            return
        
        # 提取特征
        for image_path in filtered_paths:
            clip_features, resnet_features, _ = await self.extract_features(image_path)
            clip_features_list.append(clip_features)
            resnet_features_list.append(resnet_features)
        
        clip_features_array = np.array(clip_features_list)
        resnet_features_array = np.array(resnet_features_list)
        
        # 创建或更新索引
        if self.clip_index is None or rebuild:
            # 创建新的索引
            self.clip_index = faiss.IndexFlatIP(clip_features_array.shape[1])
            self.resnet_index = faiss.IndexFlatIP(resnet_features_array.shape[1])
            self.metadata = []
            
            # 添加向量
            self.clip_index.add(clip_features_array)
            self.resnet_index.add(resnet_features_array)
            self.metadata.extend(filtered_metadata)
        else:
            # 追加向量到现有索引
            self.clip_index.add(clip_features_array)
            self.resnet_index.add(resnet_features_array)
            self.metadata.extend(filtered_metadata)

    async def search(self, image_path, clip_weight=0.6, top_k=10, min_area_ratio=0.1):
        """
        搜索相似图像
        :param image_path: 查询图像路径
        :param clip_weight: CLIP 特征的权重 (0-1)
        :param top_k: 返回结果数量
        :param min_area_ratio: 最小主要区域比例，低于此值的结果会被降权
        :return: (搜索结果列表, 热力图base64, 原图base64)
        """
        clip_features, resnet_features, area_ratio, heatmap_base64, original_base64 = await self.extract_features(image_path)
        
        # 分别使用 CLIP 和 ResNet 特征进行搜索
        clip_scores, clip_indices = self.clip_index.search(np.array([clip_features]), top_k)
        resnet_scores, resnet_indices = self.resnet_index.search(np.array([resnet_features]), top_k)
        
        # 创建一个字典来存储每个药材的最佳分数
        results_dict = {}
        
        # 根据区域比例调整权重
        area_weight = min(1.0, area_ratio / min_area_ratio)
        
        # 处理所有结果
        for i in range(len(clip_indices[0])):
            idx = int(clip_indices[0][i])
            clip_score = float(clip_scores[0][i])
            herb_name = self.metadata[idx]['name']
            
            # 找到对应的 ResNet 分数
            resnet_score = 0.0
            for j in range(len(resnet_indices[0])):
                if int(resnet_indices[0][j]) == idx:
                    resnet_score = float(resnet_scores[0][j])
                    break
            
            # 计算加权分数，并应用区域权重
            total_score = (clip_weight * clip_score + (1 - clip_weight) * resnet_score) * area_weight
            
            # 如果这个药材还没有添加到结果中，或者新的分数更高，则更新结果
            if herb_name not in results_dict or total_score > results_dict[herb_name]['total_score']:
                results_dict[herb_name] = {
                    'metadata': self.metadata[idx],
                    'total_score': total_score,
                    'clip_score': clip_score,
                    'resnet_score': resnet_score,
                    'area_ratio': area_ratio
                }
        
        # 处理剩余的 ResNet 结果
        for i in range(len(resnet_indices[0])):
            idx = int(resnet_indices[0][i])
            herb_name = self.metadata[idx]['name']
            
            # 如果这个药材已经在结果中，跳过
            if herb_name in results_dict:
                continue
                
            resnet_score = float(resnet_scores[0][i])
            
            # 找到对应的 CLIP 分数
            clip_score = 0.0
            for j in range(len(clip_indices[0])):
                if int(clip_indices[0][j]) == idx:
                    clip_score = float(clip_scores[0][j])
                    break
            
            # 计算加权分数，并应用区域权重
            total_score = (clip_weight * clip_score + (1 - clip_weight) * resnet_score) * area_weight
            
            # 将结果添加到字典中
            results_dict[herb_name] = {
                'metadata': self.metadata[idx],
                'total_score': total_score,
                'clip_score': clip_score,
                'resnet_score': resnet_score,
                'area_ratio': area_ratio
            }
        
        # 将字典转换为列表并排序
        results = list(results_dict.values())
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 打印调试信息
        print(f"Total results before top_k: {len(results)}")
        print(f"First result score: {results[0]['total_score'] if results else None}")
        
        return results[:top_k], heatmap_base64, original_base64

    def save_index(self):
        """
        保存索引和元数据
        """
        # 保存 FAISS 索引
        faiss.write_index(self.clip_index, self.index_path + '.clip')
        faiss.write_index(self.resnet_index, self.index_path + '.resnet')
        
        # 保存元数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load_index(self):
        """
        加载索引和元数据
        """
        # 加载 FAISS 索引
        self.clip_index = faiss.read_index(self.index_path + '.clip')
        self.resnet_index = faiss.read_index(self.index_path + '.resnet')
        
        # 加载元数据
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f) 