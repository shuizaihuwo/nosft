# summarization_model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class SummarizationModel:
    def __init__(self, config):
        # 从配置中加载摘要模型的名称或路径
        self.model_name = config.summarization_model_name
        # 加载预训练的摘要模型和对应的tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        # 设置设备（CPU或GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def summarize_text(self, text):
        # 对输入文本进行编码，限制最大长度以适应模型
        inputs = self.tokenizer([text], return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 使用模型生成摘要
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        # 解码生成的摘要文本
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
