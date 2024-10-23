# retrieval/retrieval_engine.py

from lightrag import LightRAG
from models.embedding_model import EmbeddingModel
from utils.config import Config
from utils.logger import Logger
import os

class RetrievalEngine:
    def __init__(self, config):
        # 初始化日志器
        self.logger = Logger.get_logger(__name__)
        self.config = config
        
        # 初始化嵌入模型
        self.embedding_model = EmbeddingModel(self.config)
        
        # 初始化LightRAG实例并加载索引
        self.rag = LightRAG(
            embedding_model=self.embedding_model,
            index_path=os.path.join(self.config.data_dir, 'retrieval', 'vector_store', 'lightrag_index'),
            top_k=self.config.top_k
        )
        self.rag.load_index()
        self.logger.info("加载LightRAG索引完成。")
    
    def retrieve_documents(self, query_embedding, top_k=None):
        # 如果未指定top_k，使用配置中的top_k
        if top_k is None:
            top_k = self.config.top_k
        
        # 使用LightRAG检索相关文档
        self.logger.debug("开始检索相关文档...")
        retrieved_docs = self.rag.retrieve(query_embedding, top_k=top_k)
        
        # 合并检索到的文档内容
        retrieved_text = "\n".join([doc['text'] for doc in retrieved_docs])
        self.logger.debug(f"检索到的文档数量: {len(retrieved_docs)}")
        
        return retrieved_text
