from pathlib import Path
import json

class HerbService:
    def __init__(self, metadata_path="static/herb_metadata.json"):
        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """加载药材元数据"""
        if not self.metadata_path.exists():
            return {}
            
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 检查数据类型，如果是列表，转换为字典
            if isinstance(data, list):
                # 假设每个元素都有id字段
                return {item.get('id', str(i)): item for i, item in enumerate(data)}
            return data
            
    def get_herb_info(self, herb_id):
        """获取药材信息"""
        # 如果metadata是字典
        if isinstance(self.metadata, dict):
            return self.metadata.get(herb_id)
        # 如果metadata是列表
        elif isinstance(self.metadata, list):
            for herb in self.metadata:
                if herb.get('id') == herb_id:
                    return herb
        return None
        
    def get_model_path(self, herb_id):
        """获取3D模型路径"""
        herb_info = self.get_herb_info(herb_id)
        if not herb_info:
            return None
        # 返回3D模型路径，使用images/3d目录
        return f"{herb_id}.glb"
        
    def get_all_herbs(self):
        """获取所有药材信息"""
        return self.metadata
        
    def search_herbs(self, query):
        """搜索药材(简单的关键词匹配)"""
        results = []
        for herb_id, info in self.metadata.items():
            if (query in info['name'] or 
                query in info['pinyin'] or 
                query in info['description']):
                results.append({
                    'id': herb_id,
                    **info
                })
        return results
        
    def get_herb_properties(self, herb_id):
        """获取药材属性"""
        herb_info = self.get_herb_info(herb_id)
        if not herb_info:
            return None
        return herb_info.get('properties', {}) 