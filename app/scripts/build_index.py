import os
import json
import asyncio
from pathlib import Path
from ..services.search_service import SearchService

async def build_search_index():
    """
    构建搜索索引
    """
    # 初始化路径
    base_dir = Path(__file__).parent.parent
    static_dir = base_dir / 'static'
    images_dir = static_dir / 'images'
    metadata_path = static_dir / 'herb_metadata.json'
    index_path = static_dir / 'search_index'

    # 确保目录存在
    images_dir.mkdir(parents=True, exist_ok=True)

    # 加载药材元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 准备图片路径和元数据
    image_paths = []
    metadata_list = []
    
    for herb in metadata:
        image_path = images_dir / f"{herb['id']}.jpg"
        if image_path.exists():
            image_paths.append(str(image_path))
            metadata_list.append(herb)

    # 初始化搜索服务
    search_service = SearchService(str(index_path), str(metadata_path))

    # 添加向量(重建索引)
    print(f"开始处理 {len(image_paths)} 张图片...")
    await search_service.add_vectors(image_paths, metadata_list, rebuild=True)

    # 保存索引
    search_service.save_index()
    print("索引构建完成!")

if __name__ == '__main__':
    asyncio.run(build_search_index()) 