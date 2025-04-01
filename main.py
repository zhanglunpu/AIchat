from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import numpy as np
import cv2
from pathlib import Path
import uuid
from app.services.feature_service import FeatureExtractor
from app.services.herb_service import HerbService
from app.services.text_search_service import TextSearchService
from typing import Optional, List
from app.services.ai_service import AIService

app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板目录
templates = Jinja2Templates(directory="app/templates")

# 初始化服务
feature_extractor = FeatureExtractor()
herb_service = HerbService()
text_search = TextSearchService()
ai_service = AIService()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search/image")
async def search_image(
    request: Request,
    file: UploadFile = File(...),
    min_area_ratio: float = Form(0.1),
    response_type: str = Form("html")  # 新增参数，默认返回html
):
    # 创建临时目录
    temp_dir = Path("static/temp")
    temp_dir.mkdir(exist_ok=True)
    
    # 保存上传的图片
    temp_path = temp_dir / file.filename
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # 提取特征并搜索
    results, heatmap_base64, original_base64 = await feature_extractor.search_image(str(temp_path), min_area_ratio=min_area_ratio)
    
    # 删除临时文件
    temp_path.unlink()
    
    # 确保base64字符串正确编码
    if not heatmap_base64.startswith('data:image/jpeg;base64,'):
        heatmap_base64 = f"data:image/jpeg;base64,{heatmap_base64}"
    if not original_base64.startswith('data:image/jpeg;base64,'):
        original_base64 = f"data:image/jpeg;base64,{original_base64}"
    
    # 转换结果格式
    formatted_results = []
    for result in results:
        herb_info = result['metadata']
        formatted_result = {
            **herb_info,
            'score': float(result['total_score']),
            'clip_score': float(result['clip_score']),
            'resnet_score': float(result['resnet_score']),
            'area_ratio': float(result['area_ratio']),
            'heatmap_url': heatmap_base64,
            'original_url': original_base64
        }
        formatted_results.append(formatted_result)
    
    # 根据响应类型返回不同格式的响应
    if response_type.lower() == "json":
        return JSONResponse(content={"herbs": formatted_results})
    else:
        # 返回HTML片段
        return templates.TemplateResponse(
            "search_results.html",
            {
                "request": request,
                "herbs": formatted_results,
                "area_ratio": formatted_results[0]['area_ratio'] if formatted_results else 0
            }
        )

@app.get("/search/text")
async def search_text(
    request: Request, 
    query: str,
    response_type: Optional[str] = Query("html")  # 新增参数，默认返回html
):
    # 先尝试直接搜索药材
    herbs = herb_service.search_herbs(query)
    if not herbs:
        # 如果没有直接匹配,使用文本搜索
        results = text_search.search(query)
        # 添加药材信息
        herbs = []
        for result in results:
            herb_info = herb_service.get_herb_info(result['id'])
            herbs.append({**herb_info, **result})
    
    # 根据请求类型返回不同的响应
    if response_type.lower() == "json":
        return JSONResponse(content={"herbs": herbs})
    else:
        # 返回HTML片段
        return templates.TemplateResponse(
            "search_results.html",
            {
                "request": request,
                "herbs": herbs
            }
        )

@app.get("/test3d", response_class=HTMLResponse)
async def test_3d(request: Request):
    return templates.TemplateResponse("test_3d.html", {"request": request})

@app.post("/api/ai/query")
async def ai_query(
    request: Request,
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    处理AI问药查询
    支持文本查询或图片识别
    """
    try:
        # 打印请求信息
        print(f"收到AI问药请求，查询文本: {query}, 是否包含图片: {image is not None}")
        
        # 如果有图片，读取图片数据
        image_data = None
        if image:
            image_data = await image.read()
        
        # 如果既没有文本查询也没有图片，返回错误
        if not query and not image_data:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "请提供文本查询或上传中药材图片"}
            )
        
        # 处理查询
        result = await ai_service.process_query(query, image_data)
        
        # 返回结果
        if result["success"]:
            return JSONResponse(content=result)
        else:
            return JSONResponse(
                status_code=400,
                content=result
            )
    
    except Exception as e:
        print(f"处理AI问药请求时出错: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "服务器内部错误，请稍后再试"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)