# prompt_builder.py

class PromptBuilder:
    def __init__(self, config):
        # 初始化配置
        self.config = config
        # 初始化各部分提示的权重
        self.weights = {
            'retrieved_knowledge': self.config.weight_retrieved_knowledge,
            'prompt2': self.config.weight_prompt2,
            'prompt1': self.config.weight_prompt1
        }

    def load_base_prompt(self):
        # 加载基础提示（Prompt1），可以从文件或配置中加载
        try:
            with open('prompts/templates/base_prompt.txt', 'r', encoding='utf-8') as f:
                base_prompt = f.read()
            return base_prompt
        except FileNotFoundError:
            # 如果文件不存在，可以使用默认的基础提示
            base_prompt = "以下是一些基础信息：\n"
            return base_prompt

    def build_system_prompt(self, retrieved_knowledge, prompt2, prompt1, delta1):
        # 根据delta1调整Prompt2的权重
        if delta1 > self.config.delta1_threshold:
            # 当delta1较高时，增加Prompt2的权重
            self.weights['prompt2'] += self.config.weight_adjustment
        else:
            # 当delta1较低时，重置Prompt2的权重为默认值
            self.weights['prompt2'] = self.config.weight_prompt2

        # 构建系统提示词（System Prompt）
        system_prompt = self.config.system_role + "\n\n"

        # 添加检索到的知识（根据权重决定是否包含）
        if self.weights['retrieved_knowledge'] > 0 and retrieved_knowledge:
            system_prompt += "请参考以下专业资料回答用户的问题：\n"
            system_prompt += retrieved_knowledge + "\n\n"

        # 添加上一轮的摘要Prompt2（根据权重决定是否包含）
        if self.weights['prompt2'] > 0 and prompt2:
            system_prompt += "请注意避免重复之前的错误：\n"
            system_prompt += prompt2 + "\n\n"

        # 添加基础提示Prompt1（根据权重决定是否包含）
        if self.weights['prompt1'] > 0 and prompt1:
            system_prompt += "以下是一些可能有用的基础信息：\n"
            system_prompt += prompt1 + "\n\n"

        return system_prompt

    def adjust_prompt_weights(self, negative_weight=False, reinforce_retrieved_knowledge=False):
        # 根据需要调整提示的权重
        if negative_weight:
            # 增加Prompt2的负权重（降低其影响）
            self.weights['prompt2'] -= self.config.weight_adjustment
            if self.weights['prompt2'] < 0:
                self.weights['prompt2'] = 0  # 确保权重不为负数

        if reinforce_retrieved_knowledge:
            # 增加检索到的知识的权重
            self.weights['retrieved_knowledge'] += self.config.weight_adjustment
            # 可根据需要限制最大权重
