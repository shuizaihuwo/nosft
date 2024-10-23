# embedding_model.py

from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, config):
        # 从配置中加载嵌入模型的名称或路径
        self.model_name = config.embedding_model_name
        # 加载预训练的SentenceTransformer模型
        self.model = SentenceTransformer(self.model_name)
    
    def get_embedding(self, text):
        # 使用SentenceTransformer模型生成文本的嵌入向量
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
