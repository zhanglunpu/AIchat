import torch
import torch.nn as nn
from pathlib import Path

# 定义药材类别
HERB_CLASSES = [
    'jinyinhua',  # 金银花
    'huaihua',    # 槐花
    'gouqi',      # 枸杞
    'dangshen',   # 党参
    'baihe'       # 百合
]

class HerbModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的ResNet50作为基础模型
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        # 添加新的分类层
        self.classifier = nn.Linear(2048, len(HERB_CLASSES))
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def get_model():
    model = HerbModel()
    model_path = Path("app/models/herb_model.pth")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
    return model 