# rag_gradio_app.py

import os
import json
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np
import faiss
import jieba
import gradio as gr
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

# -------------------- 数据结构定义 --------------------

@dataclass
class Document:
    id: str
    title: str
    content: str
    source: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class QueryResult:
    query: str
    answer: str
    source_documents: List[Document]
    confidence_score: float
    retrieval_time: float
    generation_time: float

# -------------------- 文档处理模块 --------------------

class DocumentProcessor:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:""\'\(\)（）【】\-]', '', text)
        return text.strip()

    def split_text_by_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[。！？；\n]', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> List[str]:
        sentences = self.split_text_by_sentences(text)
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def process_document(self, content: str, title: str, source: str, metadata: Dict = None) -> List[Document]:
        if metadata is None:
            metadata = {}
        clean_content = self.clean_text(content)
        chunks = self.create_chunks(clean_content)
        return [Document(
            id=f"{source}_{i}", title=title, content=chunk, source=source,
            metadata={**metadata, 'chunk_index': i, 'total_chunks': len(chunks)})
            for i, chunk in enumerate(chunks)
        ]

# -------------------- 向量构建模块 --------------------

class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_texts([query])[0]

# -------------------- 向量库管理模块 --------------------

class VectorStore:
    def __init__(self, embedding_dim: int, index_path="vector.index", docs_path="documents.pkl"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.docs_path = docs_path
        
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, "rb") as f:
                self.documents = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.documents = []

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_documents(self, docs: List[Document], embeddings: np.ndarray):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for i, doc in enumerate(docs):
            doc.embedding = embeddings[i]
        self.documents.extend(docs)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k=5) -> List[Tuple[Document, float]]:
        if len(self.documents) == 0:
            return []  # 向量库为空直接返回空列表

        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        return [
            (self.documents[idx], float(scores[0][i]))
            for i, idx in enumerate(indices[0])
            if 0 <= idx < len(self.documents)
        ]

    def list_documents(self, max_chars=100) -> List[str]:
        return [f"[{i}] {doc.title} - {doc.content[:max_chars]}..." for i, doc in enumerate(self.documents)]

    def search_by_keyword(self, keyword: str, max_results=10) -> List[str]:
        results = []
        for i, doc in enumerate(self.documents):
            if keyword in doc.content or keyword in doc.title:
                results.append(f"[{i}] {doc.title} - {doc.content[:100]}...")
            if len(results) >= max_results:
                break
        return results or ["未找到匹配结果。"]

    def delete_document_by_index(self, index: int) -> str:
        if 0 <= index < len(self.documents):
            del self.documents[index]
            # 重新构建索引
            embeddings = np.array([doc.embedding for doc in self.documents if doc.embedding is not None])
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if len(embeddings) > 0:
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
            self.save()
            return f"已删除第 {index} 个文档片段并重新构建索引。"
        return f"无效索引：{index}"


# -------------------- 回复生成模块 --------------------

class AnswerGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def generate_base_answer(self, query: str) -> str:
        # 大模型直接根据提问回答，无上下文
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个中文问答助手"},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def generate_enhanced_answer(self, query: str, docs: List[Document], base_answer: str) -> str:
        if not docs:
            # 没有检索到相关文档，直接返回基础回答
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


# -------------------- RAG系统 --------------------

class RAGSystem:
    def __init__(self, api_key: str):
        self.processor = DocumentProcessor()
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore(self.embedder.embedding_dim)
        self.generator = AnswerGenerator(api_key)

    def add_pdf(self, pdf_file) -> str:
        pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in pdf])
        docs = self.processor.process_document(text, title=pdf.name, source=pdf.name)
        embeddings = self.embedder.encode_texts([doc.content for doc in docs])
        self.vector_store.add_documents(docs, embeddings)
        return f"成功处理文档，共{len(docs)}个片段"

    def query(self, question: str) -> QueryResult:
        start = time.time()
        q_embed = self.embedder.encode_query(question)
        top_docs = self.vector_store.search(q_embed)
        retrieval_time = time.time() - start

        # 先调用大模型原回答
        base_answer = self.generator.generate_base_answer(question)

        start = time.time()
        enhanced_answer = self.generator.generate_enhanced_answer(question, [doc for doc, _ in top_docs], base_answer)
        generation_time = time.time() - start

        avg_score = sum(score for _, score in top_docs) / (len(top_docs) or 1)

        return QueryResult(
            query=question,
            answer=enhanced_answer,
            source_documents=[doc for doc, _ in top_docs],
            confidence_score=avg_score,
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )


# -------------------- Gradio界面 --------------------

rag = RAGSystem(api_key=os.getenv("DEEPSEEK_API_KEY", "sk-de353379ea494b98b197dce3b8ce5391"))

def upload_pdf_in_blocks(pdf_file):
    if not pdf_file:
        return "请上传有效的PDF文件。"
    import fitz
    pdf = fitz.open(pdf_file.name)
    text = "\n".join([page.get_text() for page in pdf])
    docs = rag.processor.process_document(text, title=Path(pdf_file.name).name, source=Path(pdf_file.name).name)
    embeddings = rag.embedder.encode_texts([doc.content for doc in docs])
    rag.vector_store.add_documents(docs, embeddings)
    return f"成功处理文档，共{len(docs)}个片段"

def qa_interface_in_blocks(question):
    result = rag.query(question)
    docs = "\n\n".join([f"[{i+1}] {doc.content[:200]}..." for i, doc in enumerate(result.source_documents)])
    return f"【回复】{result.answer}\n\n【来源文档片段】\n{docs}\n\n置信度: {result.confidence_score:.2f}，检索: {result.retrieval_time:.2f}s，生成: {result.generation_time:.2f}s"

with gr.Blocks() as app:
    gr.Markdown("## 基于RAG的行业知识问答系统")
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="上传PDF文档", file_types=['.pdf'])
            upload_feedback = gr.Textbox(label="上传反馈")
            upload_btn = gr.Button("上传文档")
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="请输入你的问题")
            answer_output = gr.Textbox(label="回答结果")
            ask_btn = gr.Button("提问")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 向量数据库管理")
            list_btn = gr.Button("📋 查看所有文档片段")
            list_output = gr.Textbox(label="文档列表", lines=10)

            search_input = gr.Textbox(label="关键词搜索")
            search_btn = gr.Button("🔍 搜索")
            search_output = gr.Textbox(label="搜索结果", lines=5)

            delete_input = gr.Number(label="删除文档片段编号")
            delete_btn = gr.Button("❌ 删除")
            delete_output = gr.Textbox(label="删除结果")

    upload_btn.click(upload_pdf_in_blocks, inputs=pdf_input, outputs=upload_feedback)
    ask_btn.click(qa_interface_in_blocks, inputs=question_input, outputs=answer_output)
    list_btn.click(fn=lambda: "\n".join(rag.vector_store.list_documents()),
                   outputs=list_output)

    search_btn.click(fn=lambda kw: "\n".join(rag.vector_store.search_by_keyword(kw)),
                     inputs=search_input,
                     outputs=search_output)

    delete_btn.click(fn=lambda i: rag.vector_store.delete_document_by_index(int(i)),
                     inputs=delete_input,
                     outputs=delete_output)


if __name__ == "__main__":
    app.launch()
