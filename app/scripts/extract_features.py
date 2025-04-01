import torch
from PIL import Image
import json
import os
from pathlib import Path
from torchvision import transforms
from app.services.model_service import get_model

def extract_features():
    # 加载训练好的模型
    model = get_model()
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载metadata
    metadata_path = Path("app/data/metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # 提取特征
    for herb in metadata["herbs"]:
        img_path = Path(f"app/static/images/{herb['id']}.jpg")
        if img_path.exists():
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # 提取特征
            with torch.no_grad():
                features = model.extract_features(image_tensor)
                
            # 将特征转换为列表并保存
            features_list = features.squeeze().cpu().numpy().tolist()
            herb["feature_vector"] = features_list
    
    # 保存更新后的metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    extract_features() 