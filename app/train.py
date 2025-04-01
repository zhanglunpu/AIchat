import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from app.data.dataset import create_dataloaders
from app.models.herb_model import HerbNet, train_epoch, evaluate, extract_features
from app.services.search_service import SearchService

def main():
    # 配置
    data_dir = "data/images/Chinese Medicine"  # 更新数据路径
    model_save_dir = "data/models"
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Loading data from: {data_dir}")
    
    # 创建保存目录
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    train_loader, test_loader, classes = create_dataloaders(data_dir, batch_size)
    print(f"Found {len(classes)} classes")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # 保存类别映射
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    with open(Path(model_save_dir) / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    
    # 创建模型
    model = HerbNet(num_classes=len(classes))
    model = model.to(device)
    print(f"Created model with {len(classes)} output classes")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    best_acc = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"Saving best model with accuracy: {best_acc:.2f}%")
            torch.save(model.state_dict(), Path(model_save_dir) / "best_model.pth")
            
    print("\nTraining completed!")
    print(f"Best accuracy: {best_acc:.2f}%")
            
    # 提取特征并构建索引
    print("\nExtracting features for search index...")
    model.load_state_dict(torch.load(Path(model_save_dir) / "best_model.pth"))
    features, labels = extract_features(model, train_loader, device)
    
    # 创建元数据
    print("Creating metadata...")
    metadata = []
    for idx, (path, label) in enumerate(train_loader.dataset.samples):
        metadata.append({
            "id": str(idx),
            "name": classes[label],
            "image_url": str(path),
            "description": f"{classes[label]}的图片"  # 这里可以添加更详细的描述
        })
    
    # 初始化搜索服务
    print("Building search index...")
    search_service = SearchService()
    search_service.init_index(features.shape[1])
    search_service.add_vectors(features.numpy(), metadata)
    
    # 保存索引和元数据
    print("Saving index and metadata...")
    search_service.save_index(
        Path(model_save_dir) / "image.index",
        Path(model_save_dir) / "metadata.json"
    )
    
    print("All done!")
    
if __name__ == "__main__":
    main() 