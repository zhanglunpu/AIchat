import torch
import torch.nn as nn
import torchvision.models as models

class HerbNet(nn.Module):
    def __init__(self, num_classes=163, pretrained=True):
        super().__init__()
        # 加载预训练的ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # 获取特征提取器(除最后一层外的所有层)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 特征维度
        self.feature_dim = self.resnet.fc.in_features
        
        # 分类头
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # 提取特征
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        # 分类
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
        
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    return test_loss / len(test_loader), 100. * correct / total

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            _, batch_features = model(data, return_features=True)
            features.append(batch_features.cpu())
            labels.append(target)
            
    return torch.cat(features), torch.cat(labels) 