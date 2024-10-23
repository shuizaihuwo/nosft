# similarity_metrics.py

import numpy as np

class SimilarityMetrics:
    def __init__(self, config):
        # 可以根据需要从配置中加载参数
        pass

    def calculate_similarity(self, embedding1, embedding2):
        # 计算两个嵌入向量之间的余弦相似度

        # 确保嵌入向量是numpy数组
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        # 计算向量的点积
        dot_product = np.dot(embedding1, embedding2)
        # 计算向量的范数
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        # 防止除以零
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)
        return similarity
