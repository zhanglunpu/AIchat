from transformers import BertModel, BertTokenizer
import torch
import numpy as np

class TextEncoder:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
    async def encode_text(self, text):
        """将文本编码为向量"""
        # 对文本进行分词
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的输出作为文本表示
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings.cpu().numpy()
        
    def __call__(self, text):
        return self.encode_text(text)
        
class TextSearchService:
    def __init__(self, encoder=None):
        self.encoder = encoder or TextEncoder()
        self.image_search_service = None
        
    def set_image_search_service(self, service):
        """设置图像搜索服务"""
        self.image_search_service = service
        
    async def search(self, query, k=5):
        """通过文本查询相似图片"""
        if self.image_search_service is None:
            raise ValueError("图像搜索服务未设置")
            
        # 将查询文本转换为向量
        query_vector = await self.encoder.encode_text(query)
        
        # 使用图像搜索服务搜索相似图片
        results = await self.image_search_service.search(query_vector, k=k)
        
        return results 