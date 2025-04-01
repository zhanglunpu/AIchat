import os
import httpx
import base64
from PIL import Image
from io import BytesIO
import json
from typing import Optional, Union, Dict, Any
from app.services.mock_deepseek_service import MockDeepseekService
from app.services.image_recognition_service import ImageRecognitionService

class AIService:
    def __init__(self):
        # 直接设置DeepSeek API密钥
        self.deepseek_api_key = "sk-42a31366e57d4cff9fa9aee012b8817b"  # 替换为您的实际密钥
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # 强制使用真实API
        os.environ["USE_REAL_DEEPSEEK_API"] = "true"
        
        # 图像识别模型API配置
        self.image_recognition_url = os.getenv("IMAGE_RECOGNITION_URL", "http://localhost:8001/recognize")
        
        # 中药材知识库 - 可以从数据库或文件加载
        self.herb_knowledge = self._load_herb_knowledge()
        
        # 初始化模拟服务
        self.mock_deepseek = MockDeepseekService()
        self.image_recognition = ImageRecognitionService()
    
    def _load_herb_knowledge(self):
        """加载中药材知识库，用于构建提示词"""
        try:
            # 从文件加载中药材知识库
            with open("static/herb_metadata.json", "r", encoding="utf-8") as f:
                herbs = json.load(f)
            
            # 构建简单的知识库
            knowledge = {}
            for herb in herbs:
                knowledge[herb["name"]] = {
                    "description": herb.get("description", ""),
                    "taste": herb.get("properties", {}).get("taste", ""),
                    "nature": herb.get("properties", {}).get("nature", ""),
                    "meridians": herb.get("properties", {}).get("meridians", []),
                    "functions": herb.get("properties", {}).get("functions", [])
                }
            return knowledge
        except Exception as e:
            print(f"加载中药材知识库失败: {e}")
            return {}
    
    def _construct_prompt(self, query: str) -> str:
        """根据查询构建专业的提示词"""
        base_prompt = (
            "你是一位经验丰富的中医师，精通中草药药理、功效、配伍和使用方法。请基于专业中医知识回答以下问题，"
            "答案应包括药材特性、功效作用、使用禁忌等相关信息。如果涉及多种药材配伍，请详细说明其协同作用和可能的禁忌。"
            "回答要专业、全面但通俗易懂，适合普通人理解。\n\n问题: {query}"
        )
        return base_prompt.format(query=query)
    
    async def query_deepseek(self, prompt: str) -> str:
        """调用deepseek模型或模拟服务获取回答"""
        try:
            # 使用环境变量控制是否使用真实API
            use_real_api = os.getenv("USE_REAL_DEEPSEEK_API", "false").lower() == "true"
            
            if use_real_api:
                # 使用真实的Deepseek API - 保持原有代码
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.deepseek_api_key}"
                }
                
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一位专业的中医师，精通中草药知识。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 800
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.deepseek_api_url,
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        print(f"Deepseek API请求失败: {response.status_code}, {response.text}")
                        return "抱歉，AI服务暂时不可用，请稍后再试。"
            
            else:
                # 使用模拟服务
                return await self.mock_deepseek.generate_response(prompt)
                
        except Exception as e:
            print(f"查询Deepseek模型时出错: {e}")
            return "抱歉，处理您的问题时出现了错误，请稍后再试。"
    
    async def recognize_herb_image(self, image_data: bytes) -> Dict[str, Any]:
        """识别图片中的中药材"""
        try:
            # 使用环境变量控制是否使用真实API
            use_real_api = os.getenv("USE_REAL_RECOGNITION_API", "false").lower() == "true"
            
            if use_real_api:
                # 使用真实的图像识别API - 保持原有代码
                # 将图片转为base64编码
                encoded_image = base64.b64encode(image_data).decode("utf-8")
                
                # 调用图像识别API
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self.image_recognition_url,
                        json={"image": encoded_image}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # 如果识别成功，返回识别结果
                        if "herb" in result and result["herb"]:
                            herb_name = result["herb"]["name"]
                            
                            # 构建关于该药材的问题
                            prompt = self._construct_prompt(f"{herb_name}的功效与作用是什么？使用禁忌有哪些？")
                            
                            # 调用deepseek获取详细信息
                            ai_answer = await self.query_deepseek(prompt)
                            
                            return {
                                "success": True,
                                "herb": result["herb"],
                                "answer": ai_answer
                            }
                        else:
                            return {
                                "success": False,
                                "message": "未能识别图片中的中药材，请尝试上传更清晰的图片。"
                            }
                    else:
                        print(f"图像识别API请求失败: {response.status_code}, {response.text}")
                        return {
                            "success": False,
                            "message": "图像识别服务暂时不可用，请稍后再试。"
                        }
            
            else:
                # 使用模拟服务
                result = await self.image_recognition.recognize(image_data)
                
                if result["success"] and "herb" in result:
                    herb_name = result["herb"]["name"]
                    
                    # 构建关于该药材的问题
                    prompt = self._construct_prompt(f"{herb_name}的功效与作用是什么？使用禁忌有哪些？")
                    
                    # 调用deepseek获取详细信息
                    ai_answer = await self.query_deepseek(prompt)
                    
                    result["answer"] = ai_answer
                
                return result
                
        except Exception as e:
            print(f"识别中药材图片时出错: {e}")
            return {
                "success": False,
                "message": "处理图片时出现了错误，请稍后再试。"
            }
    
    async def process_query(self, query: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """处理用户查询，可以是文本或图片"""
        try:
            # 如果提供了图片数据，优先处理图片
            if image_data:
                return await self.recognize_herb_image(image_data)
            
            # 处理文本查询
            prompt = self._construct_prompt(query)
            answer = await self.query_deepseek(prompt)
            
            return {
                "success": True,
                "answer": answer
            }
        
        except Exception as e:
            print(f"处理查询时出错: {e}")
            return {
                "success": False,
                "message": "处理您的请求时出现了错误，请稍后再试。"
            } 