# models/llm_interface.py

import requests
import json
from utils.config import Config
from utils.logger import Logger

class LLMInterface:
    def __init__(self, config):
        # 初始化日志器
        self.logger = Logger.get_logger(__name__)
        
        # 从配置中加载SiliconFlow的API配置
        self.url = config.siliconflow_api_url  # SiliconFlow API的URL
        self.api_key = config.siliconflow_api_key  # SiliconFlow API密钥
        self.model_name = config.siliconflow_model_name  # 使用的模型名称
        
        # 设置请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, system_prompt, user_query):
        # 构建消息列表，包含系统提示词和用户输入
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # 构建请求负载
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": 512,
            "stop": ["<string>"],
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "json_object"}
        }
        
        try:
            # 发送POST请求到SiliconFlow的API
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()  # 检查请求是否成功
            
            # 解析响应内容
            response_data = response.json()
            
            # 提取生成的回复内容，根据返回的结构调整
            reply = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            return reply
        except requests.exceptions.RequestException as e:
            # 处理请求异常，记录错误日志
            self.logger.error(f"调用SiliconFlow接口时出错：{e}")
            return "抱歉，我无法处理您的请求。请稍后再试。"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            # 处理解析异常，记录错误日志
            self.logger.error(f"解析SiliconFlow响应时出错：{e}")
            return "抱歉，我无法理解来自服务器的响应。请稍后再试。"
