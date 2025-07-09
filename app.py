# app.py
"""
Gradio 主入口，整合 rag_core 的各个模块，提供上传文档、提问问答、
向量数据库查看与管理等功能的可视化界面。
"""

import os
import fitz
from pathlib import Path
import shutil
import docx  # 读取 Word 文件

import gradio as gr

from rag_core.doc_processor import DocumentProcessor
from rag_core.embedder import EmbeddingModel
from rag_core.vector_store import VectorStore
from rag_core.generator import AnswerGenerator

import os
import fitz
from pathlib import Path
# ...

import gradio as gr

# 🔽 读取外部 CSS 文件
with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()


# 初始化 RAG 组件
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-de353379ea494b98b197dce3b8ce5391")  # 你的API KEY
processor = DocumentProcessor()
embedder = EmbeddingModel()
vector_store = VectorStore(embedder.embedding_dim)
generator = AnswerGenerator(API_KEY)

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

    original_filename = Path(file.name).name

    # 判断是否已上传过相同文档
    if any(doc.source == original_filename for doc in vector_store.documents):
        return f"⚠️ 检测到已存在相同名称的文档：{original_filename}，已跳过上传。"

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

import re
import markdown

def qa_interface(question):
    q_embed = embedder.encode_query(question)
    top_docs = vector_store.search(q_embed)
    base_answer = generator.generate_base_answer(question)
    enhanced = generator.generate_enhanced_answer(question, [doc for doc, _ in top_docs], base_answer)

    confidence = sum(score for _, score in top_docs) / (len(top_docs) or 1)
    result_docs = "\n\n".join([f"[{i+1}] {doc.content[:30]}..." for i, (doc, _) in enumerate(top_docs)])

    # Markdown 转 HTML
    enhanced_html = markdown.markdown(enhanced, extensions=['tables', 'fenced_code'])

    # 文档片段做简单的换行处理为 <br>
    docs_html = result_docs.replace('\n', '<br>')

    # 构造完整 HTML 返回
    html = f"""
    <div id="custom-html-answer">
    <div><strong>【回复】</strong><br>{enhanced_html}</div>
    <div style="margin-top: 1em;"><strong>【来源文档片段】</strong><br>{docs_html}</div>
    <div style="margin-top: 1em;"><strong>置信度:</strong> {confidence:.2f}</div>
    </div>
    """


    return html


def list_docs():
    return "\n".join(vector_store.list_documents())

def search_by_keyword(keyword):
    return "\n".join(vector_store.search_by_keyword(keyword))

def delete_by_index(index):
    try:
        return vector_store.delete_document_by_index(int(index))
    except:
        return "❌ 输入的编号无效，请输入整数。"

def batch_delete_by_indices(indices_str: str):
    import re

    indices = set()
    try:
        parts = [p.strip() for p in indices_str.split(",")]
        for part in parts:
            if re.match(r"^\d+-\d+$", part):
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

    results = []
    for index in sorted(indices, reverse=True):
        results.append(vector_store.delete_document_by_index(index))

    return "\n".join(results)

def show_thinking():
    return "<p style='font-size:28px; font-weight:bold; color:#666;'>🤔 结合数据库搜索，深度思考中......</p>"


with gr.Blocks(css=css) as app:
    with gr.Tabs():
        # ✅ 问答页面
        with gr.TabItem("💬 问答页面"):
            with gr.Column(scale=3):
                # 🔹 回复展示区域（滚动框）
                with gr.Group(elem_classes="fixed-box"):
                    answer_output = gr.HTML(elem_id="custom-html-answer")

                # 🔹 输入框 + 提问按钮 横向排列
                with gr.Row():
                    with gr.Column(scale=9):
                        question_input = gr.Textbox(label="❓ 请输入问题", elem_id="question-box")
                    with gr.Column(scale=1):
                        ask_btn = gr.Button("🚀 发送", elem_id="ask-button", elem_classes="qa-button")


            # 🔹 问答逻辑绑定
            ask_btn.click(show_thinking, inputs=None, outputs=answer_output)
            question_input.submit(show_thinking, inputs=None, outputs=answer_output)

            ask_btn.click(qa_interface, inputs=question_input, outputs=answer_output, queue=True)
            question_input.submit(qa_interface, inputs=question_input, outputs=answer_output, queue=True)

        # ✅ 数据库管理页面
        with gr.TabItem("📁 数据库管理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 文档上传")
                    pdf_input = gr.File(label="📎 上传文档", file_types=[".pdf", ".docx", ".txt"])
                    upload_btn = gr.Button("📤 确认上传", elem_classes="db-button")
                    upload_feedback = gr.Textbox(lines=7,label="📩 上传反馈", interactive=False)

                with gr.Column(scale=3):
                    gr.Markdown("### 📋 文档列表")
                    list_btn = gr.Button("🔄 刷新列表", elem_classes="db-button")
                    list_output = gr.Textbox(label="📚 当前文档片段", lines=3, interactive=False)

                    gr.Markdown("### 🔍 数据库关键词搜索")
                    search_input = gr.Textbox(label="📝 请输入要检索的文档关键词：")
                    search_btn = gr.Button("🔍 搜索", elem_classes="db-button")
                    search_output = gr.Textbox(label="📄 检索结果", lines=3, interactive=False)

                with gr.Column(scale=3):
                    gr.Markdown("### 🗑️ 删除文档片段")
                    delete_input = gr.Textbox(label="✂️ 删除编号（支持单个、逗号分隔、区间，如：3、1,4、2-5）")
                    delete_btn = gr.Button("🗑️ 删除", elem_classes="db-button")
                    delete_output = gr.Textbox(lines=14,label="🧾 删除结果", interactive=False)

    # ✅ 功能绑定
    upload_btn.click(upload_document, inputs=pdf_input, outputs=upload_feedback)
    list_btn.click(list_docs, outputs=list_output)
    search_btn.click(search_by_keyword, inputs=search_input, outputs=search_output)
    delete_btn.click(batch_delete_by_indices, inputs=delete_input, outputs=delete_output)


if __name__ == "__main__":
    app.launch()
