# app.py
"""
Gradio 主入口，整合 rag_core 的各个模块，提供上传文档、提问问答、
向量数据库查看与管理等功能的可视化界面。
"""

import os
import fitz
from pathlib import Path
import gradio as gr

from rag_core.doc_processor import DocumentProcessor
from rag_core.embedder import EmbeddingModel
from rag_core.vector_store import VectorStore
from rag_core.generator import AnswerGenerator
from rag_core.data_model import QueryResult

# 初始化 RAG 组件
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-de353379ea494b98b197dce3b8ce5391")  # 默认可以修改成你的 key
processor = DocumentProcessor()
embedder = EmbeddingModel()
vector_store = VectorStore(embedder.embedding_dim)
generator = AnswerGenerator(API_KEY)

import shutil

import docx  # 用于读取 Word 文件

def upload_document(file):
    if not file:
        return "❌ 请上传有效的文件。"

    ext = Path(file.name).suffix.lower()
    if ext == ".pdf":
        sub_dir = "pdf"
    elif ext == ".docx":
        sub_dir = "docx"
    elif ext == ".txt":
        sub_dir = "txt"
    else:
        return "❌ 不支持的文件类型（仅支持 PDF / DOCX / TXT）"

    # 提取文件名
    original_filename = Path(file.name).name

    # ✅ 检查是否已上传过相同的文档（按 source 字段判断）
    if any(doc.source == original_filename for doc in vector_store.documents):
        return f"⚠️ 检测到已存在相同名称的文档：{original_filename}，已跳过上传。"

    # 保存文件到子目录
    save_dir = os.path.join("raw_data", sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, original_filename)
    shutil.copyfile(file.name, save_path)

    # 读取文件内容
    if ext == ".pdf":
        doc = fitz.open(file.name)
        text = "\n".join([page.get_text() for page in doc])
    elif ext == ".docx":
        text = "\n".join([para.text for para in docx.Document(file.name).paragraphs])
    elif ext == ".txt":
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()

    docs = processor.process_document(text, title=original_filename, source=original_filename)
    embeddings = embedder.encode_texts([doc.content for doc in docs])
    vector_store.add_documents(docs, embeddings)
    return f"✅ 成功上传并解析 {ext[1:].upper()} 文件，共 {len(docs)} 个片段"



# 功能：问答
def qa_interface(question):
    q_embed = embedder.encode_query(question)
    top_docs = vector_store.search(q_embed)
    base_answer = generator.generate_base_answer(question)
    enhanced = generator.generate_enhanced_answer(question, [doc for doc, _ in top_docs], base_answer)
    confidence = sum(score for _, score in top_docs) / (len(top_docs) or 1)
    result_docs = "\n\n".join([f"[{i+1}] {doc.content[:30]}..." for i, (doc, _) in enumerate(top_docs)])
    return f"""【回复】{enhanced}\n\n【来源文档片段】\n{result_docs}\n\n置信度: {confidence:.2f}"""

# 功能：列出所有文档
def list_docs():
    return "\n".join(vector_store.list_documents())

# 功能：关键词检索
def search_by_keyword(keyword):
    return "\n".join(vector_store.search_by_keyword(keyword))

# 功能：删除指定文档片段
def delete_by_index(index):
    try:
        return vector_store.delete_document_by_index(int(index))
    except:
        return "❌ 输入的编号无效，请输入整数。"

# 功能：批量删除    
def batch_delete_by_indices(indices_str: str):
    """
    支持格式示例：
    "1,3,5"           -> 删除索引1,3,5
    "1-5"             -> 删除索引1到5（含）
    "1,3-5,7"         -> 删除索引1，3到5，以及7
    """
    import re

    indices = set()
    try:
        parts = [p.strip() for p in indices_str.split(",")]
        for part in parts:
            if re.match(r"^\d+-\d+$", part):  # 区间格式
                start, end = map(int, part.split("-"))
                if start > end:
                    return "❌ 区间格式错误，起始索引应小于等于结束索引"
                indices.update(range(start, end + 1))
            elif part.isdigit():
                indices.add(int(part))
            else:
                return f"❌ 输入格式错误，不能解析部分：'{part}'"
    except Exception as e:
        return f"❌ 解析输入时发生错误: {e}"

    # 逆序删除避免索引错乱
    results = []
    for index in sorted(indices, reverse=True):
        results.append(vector_store.delete_document_by_index(index))

    return "\n".join(results)



# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 💬 基于RAG的行业知识问答系统")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="📎 上传文档", file_types=[".pdf", ".docx", ".txt"])
            upload_btn = gr.Button("上传")
            upload_feedback = gr.Textbox(label="上传反馈")
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="❓ 请输入问题")
            ask_btn = gr.Button("提问")
            answer_output = gr.Textbox(label="系统回答", lines=12)

    with gr.Row():
        gr.Markdown("### 🧠 向量数据库管理")
    with gr.Row():
        with gr.Column():
            list_btn = gr.Button("📋 查看所有文档片段")
            list_output = gr.Textbox(label="文档列表", lines=10)

            search_input = gr.Textbox(label="🔍 关键词搜索")
            search_btn = gr.Button("搜索")
            search_output = gr.Textbox(label="搜索结果", lines=5)

            delete_input = gr.Number(label="❌ 删除文档片段编号")
            delete_btn = gr.Button("删除")
            delete_output = gr.Textbox(label="删除结果")

    with gr.Row():
        with gr.Column():
            batch_delete_input = gr.Textbox(label="❌ 批量删除文档片段编号（支持逗号分隔和区间，如1,3-5,7）")
            batch_delete_btn = gr.Button("批量删除")
            batch_delete_output = gr.Textbox(label="批量删除结果", lines=6)

    # 所有按钮绑定事件
    upload_btn.click(upload_document, inputs=pdf_input, outputs=upload_feedback)
    ask_btn.click(qa_interface, inputs=question_input, outputs=answer_output)
    list_btn.click(list_docs, outputs=list_output)
    search_btn.click(search_by_keyword, inputs=search_input, outputs=search_output)
    delete_btn.click(delete_by_index, inputs=delete_input, outputs=delete_output)
    batch_delete_btn.click(batch_delete_by_indices, inputs=batch_delete_input, outputs=batch_delete_output)


if __name__ == "__main__":
    app.launch()
