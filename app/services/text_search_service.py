import json
from pathlib import Path
from typing import List, Dict

class TextSearchService:
    def __init__(self):
        # 加载metadata
        metadata_path = Path("app/data/metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        基于文本相似度搜索药材
        
        参数:
            query: 搜索查询
            top_k: 返回结果数量
            
        返回:
            包含搜索结果的列表,每个结果包含id和相似度分数
        """
        query = query.lower()
        scores = []
        
        for herb in self.metadata["herbs"]:
            score = self._calculate_similarity(query, herb)
            scores.append((herb["id"], score))
        
        # 按分数排序并返回前top_k个结果
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"id": id, "score": score} for id, score in scores[:top_k]]
    
    def _calculate_similarity(self, query: str, herb: Dict) -> float:
        """
        计算查询和药材之间的相似度分数
        
        参数:
            query: 搜索查询
            herb: 药材信息
            
        返回:
            相似度分数 (0-1)
        """
        score = 0.0
        
        # 检查名称匹配
        if query in herb["name"].lower():
            score += 1.0
        if query in herb["pinyin"].lower():
            score += 0.8
            
        # 检查描述匹配
        if query in herb["description"].lower():
            score += 0.6
            
        # 检查功效匹配
        for func in herb["properties"]["functions"]:
            if query in func.lower():
                score += 0.4
                break
                
        # 检查归经匹配
        for meridian in herb["properties"]["meridians"]:
            if query in meridian.lower():
                score += 0.3
                break
                
        # 检查性味匹配
        if query in herb["properties"]["nature"].lower():
            score += 0.2
        if query in herb["properties"]["taste"].lower():
            score += 0.2
            
        return min(score, 1.0)  # 将分数限制在0-1之间 