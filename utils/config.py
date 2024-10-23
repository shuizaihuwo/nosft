# utils/config.py

import os

class Config:
    def __init__(self):
        # 基础目录配置
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.join(self.base_dir, '..')
        self.data_dir = os.path.join(self.project_dir, 'data')
        self.processed_data_dir = os.path.join(self.data_dir, 'processed_data')
        
        # 嵌入模型配置
        self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # 嵌入模型的名称或路径

        # LLM接口配置（SiliconFlow）
        self.siliconflow_api_url = "https://api.siliconflow.cn/v1/chat/completions"  # SiliconFlow API的URL
        self.siliconflow_api_key = "YOUR_SILICONFLOW_API_KEY"  # 请在此处填写您的SiliconFlow API密钥
        self.siliconflow_model_name = "deepseek-ai/DeepSeek-V2.5"  # 使用的SiliconFlow模型名称
        
        # 摘要模型配置
        self.summarization_model_name = 'facebook/bart-large-cnn'  # 摘要模型的名称或路径

        # 检索引擎配置
        self.top_k = 5  # 每次检索返回的文档数量

        # 相似度阈值配置
        self.delta1_threshold = 0.8  # 查询相似度阈值
        self.delta2_threshold = 0.8  # 输出相似度阈值

        # 提示权重配置
        self.weight_retrieved_knowledge = 1.0  # 检索到的知识的初始权重
        self.weight_prompt2 = 0.5  # Prompt2的初始权重
        self.weight_prompt1 = 0.3  # Prompt1的初始权重
        self.weight_adjustment = 0.1  # 权重调整的幅度

        # 系统角色设定
        self.system_role = "您是一位专业的助手，能够为用户提供准确而详细的信息。"

        # 其他配置参数可以根据需要添加
