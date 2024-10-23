# main.py

from models.llm_interface import LLMInterface
from models.embedding_model import EmbeddingModel
from models.summarization_model import SummarizationModel
from retrieval.retrieval_engine import RetrievalEngine
from prompts.prompt_builder import PromptBuilder
from utils.similarity_metrics import SimilarityMetrics
from utils.config import Config
from utils.logger import Logger

class Chatbot:
    def __init__(self):
        # 初始化日志器
        self.logger = Logger.get_logger(__name__)
        self.logger.info("初始化聊天机器人...")

        # 加载配置
        self.config = Config()

        # 初始化各个模型和组件
        self.logger.info("加载嵌入模型...")
        self.embedding_model = EmbeddingModel(self.config)

        self.logger.info("加载摘要模型...")
        self.summarization_model = SummarizationModel(self.config)

        self.logger.info("初始化LLM接口...")
        self.llm_interface = LLMInterface(self.config)

        self.logger.info("初始化检索引擎...")
        self.retrieval_engine = RetrievalEngine(self.config, self.embedding_model)

        self.logger.info("初始化提示词生成器...")
        self.prompt_builder = PromptBuilder(self.config)

        self.logger.info("初始化相似度计算器...")
        self.similarity_metrics = SimilarityMetrics(self.config)

        # 初始化变量
        self.previous_query_embedding = None
        self.previous_output_embedding = None
        self.expanded_output = ""
        self.prompt1 = self.prompt_builder.load_base_prompt()
        self.prompt2 = ""
        self.prompt3 = None  # 上一轮Expanded Output的嵌入

    def process_query(self, user_query):
        self.logger.info("处理用户查询...")

        # 计算当前查询的嵌入
        current_query_embedding = self.embedding_model.get_embedding(user_query)
        self.logger.debug(f"当前查询嵌入：{current_query_embedding}")

        # 计算δ1（查询相似度）
        if self.previous_query_embedding is not None:
            delta1 = self.similarity_metrics.calculate_similarity(current_query_embedding, self.previous_query_embedding)
            self.logger.info(f"δ1（查询相似度）={delta1}")
        else:
            delta1 = 0
            self.logger.info("这是第一轮对话，δ1设为0")

        # 检索相关知识
        retrieved_knowledge = self.retrieval_engine.retrieve_documents(current_query_embedding, top_k=self.config.top_k)
        self.logger.info("已检索相关知识")

        # 更新Prompt2（摘要上一轮的输出）
        if self.expanded_output:
            self.prompt2 = self.summarization_model.summarize_text(self.expanded_output)
            self.logger.info("已生成Prompt2（上一轮输出的摘要）")
        else:
            self.prompt2 = ""
            self.logger.info("没有上一轮的输出，Prompt2为空")

        # 构建系统提示词（Refined Context）
        system_prompt = self.prompt_builder.build_system_prompt(
            retrieved_knowledge=retrieved_knowledge,
            prompt2=self.prompt2,
            prompt1=self.prompt1,
            delta1=delta1
        )
        self.logger.info("已构建系统提示词")

        # 与LLM交互，获取模型输出
        model_output = self.llm_interface.generate_response(system_prompt, user_query)
        self.logger.info("已获取模型输出")

        # 输出模型的回答
        print(f"模型：{model_output}")

        # 计算δ2（输出相似度）
        current_output_embedding = self.embedding_model.get_embedding(model_output)
        if self.previous_output_embedding is not None:
            delta2 = self.similarity_metrics.calculate_similarity(current_output_embedding, self.previous_output_embedding)
            self.logger.info(f"δ2（输出相似度）={delta2}")
        else:
            delta2 = 0
            self.logger.info("这是第一轮对话，δ2设为0")

        # 根据δ1和δ2调整策略
        self.adjust_strategy(delta1, delta2)

        # 更新Expanded Output和嵌入
        self.expanded_output += " " + model_output
        # 控制Expanded Output的长度
        self.expanded_output = self.summarization_model.summarize_text(self.expanded_output)
        self.logger.info("已更新Expanded Output")

        # 更新上一轮的嵌入
        self.previous_query_embedding = current_query_embedding
        self.previous_output_embedding = current_output_embedding

    def adjust_strategy(self, delta1, delta2):
        # 根据δ1和δ2的组合，调整Prompt的权重或其他策略
        self.logger.info("根据δ1和δ2调整策略")
        # 示例实现，可根据具体需求完善
        if delta1 > self.config.delta1_threshold and delta2 > self.config.delta2_threshold:
            # 用户重复提问，回答无改进，需要调整
            self.logger.warning("用户重复提问，回答无改进，增加Prompt2的负权重")
            self.prompt_builder.adjust_prompt_weights(negative_weight=True)
        elif delta1 > self.config.delta1_threshold and delta2 < self.config.delta2_threshold:
            # 用户重复提问，回答有改进，无需调整
            self.logger.info("用户重复提问，回答有改进，无需调整")
        elif delta1 < self.config.delta1_threshold and delta2 > self.config.delta2_threshold:
            # 用户提出新问题，但回答相似，可能模型未理解，需要调整
            self.logger.warning("新问题但回答相似，可能需要调整模型提示词")
            self.prompt_builder.adjust_prompt_weights(reinforce_retrieved_knowledge=True)
        else:
            # 其他情况，无需调整
            self.logger.info("无需调整策略")

def main():
    chatbot = Chatbot()
    print("欢迎使用聊天机器人，输入您的问题，输入 'exit' 或 'quit' 结束对话。")
    while True:
        user_query = input("用户：")
        if user_query.lower() in ["exit", "quit"]:
            print("再见！")
            break
        chatbot.process_query(user_query)

if __name__ == "__main__":
    main()
