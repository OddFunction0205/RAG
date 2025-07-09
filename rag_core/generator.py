"""
生成模块：基础回答与增强回答（带文档引用）生成
"""
from openai import OpenAI
from .data_model import Document

class AnswerGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def generate_base_answer(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个中文问答助手"},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def generate_enhanced_answer(self, query: str, docs: list[Document], base_answer: str) -> str:
        if not docs:
            return base_answer

        context = "\n".join([f"[{i+1}] {doc.content}" for i, doc in enumerate(docs)])
        prompt = f"""你是专业的中文问答助手。请根据以下文档内容和大模型原回答，结合自身知识，生成一个更全面、详细、有条理的解答，并在使用文档信息时注明来源编号（如 [1]、[2]）。

要求：
- 用较长段落，逐点展开说明；
- 优先使用文档信息回答，必要时补充通识；
- 不要省略细节；
- 结构清晰、逻辑严谨；
- 保持中文书面语风格。

文档内容：
{context}

大模型原回答：
{base_answer}

问题：
{query}

请开始作答：
"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个中文问答助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
