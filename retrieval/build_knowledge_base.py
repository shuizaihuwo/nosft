# retrieval/build_knowledge_base.py

from lightrag import LightRAG
from models.embedding_model import EmbeddingModel
from utils.config import Config
from utils.logger import Logger
import os

class KnowledgeBaseBuilder:
    def __init__(self, config):
        # 初始化日志器
        self.logger = Logger.get_logger(__name__)
        self.config = config
        
        # 初始化嵌入模型
        self.embedding_model = EmbeddingModel(self.config)
        
        # 初始化LightRAG实例
        self.rag = LightRAG(
            embedding_model=self.embedding_model,
            index_path=os.path.join(self.config.data_dir, 'retrieval', 'vector_store', 'lightrag_index'),
            top_k=self.config.top_k
        )
    
    def build_knowledge_base(self):
        # 从processed_data中读取所有文档
        self.logger.info("开始构建知识库...")
        knowledge_files = os.listdir(self.config.processed_data_dir)
        for file_name in knowledge_files:
            file_path = os.path.join(self.config.processed_data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # 添加文档到LightRAG索引
                self.rag.add_document(text)
                self.logger.debug(f"添加文档到知识库: {file_name}")
        
        # 构建索引
        self.logger.info("构建LightRAG索引...")
        self.rag.build_index()
        self.logger.info("知识库构建完成。")

if __name__ == "__main__":
    config = Config()
    builder = KnowledgeBaseBuilder(config)
    builder.build_knowledge_base()
