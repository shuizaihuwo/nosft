# enhanced_lightRAG_script.py

import os
import sys
import asyncio
import requests
import numpy as np
from dotenv import load_dotenv
import time

# 添加项目根目录到 sys.path 以确保可以导入 lightrag.utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)  # 使用 insert(0, ...) 确保优先级更高

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc, Logger  # 正确导入 Logger

# 加载环境变量
load_dotenv()

# 初始化日志记录器
logger = Logger.get_logger(__name__)

WORKING_DIR = "./dickens"

# 创建工作目录（如果不存在）
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 配置重试参数
MAX_RETRIES = 5
INITIAL_DELAY = 2  # 初始延迟时间（秒）
BACKOFF_FACTOR = 2  # 指数回退因子

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    使用SiliconFlow的LLM接口生成响应，包含重试机制以处理HTTP 429错误。
    
    参数:
    - prompt: 用户的查询内容。
    - system_prompt: 系统提示词。
    - history_messages: 对话历史消息。
    - kwargs: 其他可选参数。
    
    返回:
    - 模型生成的响应字符串。
    """
    retry_count = 0
    delay = INITIAL_DELAY
    while retry_count < MAX_RETRIES:
        try:
            response = await openai_complete_if_cache(
                "Qwen/Qwen2.5-7B-Instruct",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("API_KEY"),
                base_url="https://api.siliconflow.cn/v1",
                **kwargs,
            )
            return response
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                # 处理HTTP 429错误，进行重试
                retry_after = int(e.response.headers.get("Retry-After", delay))
                logger.warning(f"HTTP 429 错误: 请求过多，等待 {retry_after} 秒后重试...")
                await asyncio.sleep(retry_after)
                retry_count += 1
                delay *= BACKOFF_FACTOR  # 指数回退
            else:
                # 处理其他请求异常
                logger.error(f"调用SiliconFlow LLM接口时出错: {e}")
                raise
    logger.error("达到最大重试次数，无法获取LLM响应。")
    return "抱歉，我无法处理您的请求。请稍后再试。"

async def embedding_func(texts: list[str]) -> np.ndarray:
    """
    使用SiliconFlow的嵌入接口生成文本嵌入，包含重试机制以处理HTTP 429错误。
    
    参数:
    - texts: 需要生成嵌入的文本列表。
    
    返回:
    - 生成的嵌入向量的NumPy数组。
    """
    retry_count = 0
    delay = INITIAL_DELAY
    while retry_count < MAX_RETRIES:
        try:
            embeddings = await openai_embedding(
                texts,
                model="BAAI/bge-m3",
                api_key=os.getenv("API_KEY"),
                base_url="https://api.siliconflow.cn/v1"
            )
            return embeddings
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                # 处理HTTP 429错误，进行重试
                retry_after = int(e.response.headers.get("Retry-After", delay))
                logger.warning(f"HTTP 429 错误: 请求过多，等待 {retry_after} 秒后重试...")
                await asyncio.sleep(retry_after)
                retry_count += 1
                delay *= BACKOFF_FACTOR  # 指数回退
            else:
                # 处理其他请求异常
                logger.error(f"调用SiliconFlow嵌入接口时出错: {e}")
                raise
    logger.error("达到最大重试次数，无法获取嵌入。")
    return np.array([])  # 返回空数组或根据需要处理

# 异步测试函数
async def test_funcs():
    """
    测试LLM和嵌入函数是否正常工作。
    """
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)

# 运行测试函数
asyncio.run(test_funcs())

# 初始化LightRAG实例
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embedding_func
    ),
)

# 插入知识库数据
with open("/home/qy/keyan/nosft/data/raw_data/2022-06-08-enriched.txt", 'r', encoding='utf-8') as f:
    rag.insert(f.read())

# 定义查询函数
async def perform_queries():
    """
    执行多种模式的查询，并打印结果。
    """
    queries = [
        ("What are the top themes in this story?", QueryParam(mode="naive")),
        ("What are the top themes in this story?", QueryParam(mode="local")),
        ("What are the top themes in this story?", QueryParam(mode="global")),
        ("What are the top themes in this story?", QueryParam(mode="hybrid")),
    ]

    for query, param in queries:
        try:
            response = rag.query(query, param=param)
            print(f"模式: {param.mode}, 响应: {response}")
        except Exception as e:
            logger.error(f"执行查询时出错: {e}")

# 执行查询
asyncio.run(perform_queries())
