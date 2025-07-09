接下来是项目说明文档 `README.md`，用于说明项目架构、功能模块及启动方法：

---

### 📘 README.md

```markdown
# 基于RAG的中文行业知识问答系统

本项目实现了一个基于文档检索增强生成（RAG, Retrieval-Augmented Generation）技术的中文问答系统，结合 PDF 文档上传、嵌入式向量检索、大语言模型生成，并通过 Gradio 提供交互式前端界面。

---

## 🏗️ 项目结构

```

.
├── app.py                    # 主程序入口，构建系统并启动前端界面
├── rag\_system/
│   ├── **init**.py
│   ├── document.py           # 文档结构与切分处理
│   ├── embedding.py          # 文本向量构建
│   ├── vector\_store.py       # 向量数据库管理与搜索
│   ├── generator.py          # 回答生成模块（调用DeepSeek API）
│   └── rag.py                # 封装整体RAG系统逻辑（文档处理 + 检索 + 生成）
├── requirements.txt
└── README.md

````

---

## ⚙️ 安装依赖

建议使用 `conda` 或 `venv` 虚拟环境运行本项目：

```bash
pip install -r requirements.txt
````

`requirements.txt` 示例内容：

```txt
torch
faiss-cpu
gradio
sentence-transformers
openai
PyMuPDF
jieba
transformers
```

---

## 🚀 启动项目

```bash
python app.py
```

运行后将启动 Gradio Web UI，一般在本地地址为：

```
http://127.0.0.1:7860
```

---

## 📂 功能模块说明

### 1. PDF上传与处理

* 支持用户上传 PDF 文档
* 使用正则和分句将文档切分为段落块
* 为每段生成文本嵌入并存入 FAISS 向量库

### 2. 问答功能（RAG）

* 接收用户提问，先检索相关文档片段
* 使用 DeepSeek-Chat 大模型生成基础回答
* 融合文档信息后进一步生成增强回答

### 3. 向量库管理界面

* 查看所有文档片段（带编号）
* 关键词搜索片段内容
* 删除指定编号文档片段

---

## 🔑 环境变量设置

需在环境变量或 `.env` 文件中设置 DeepSeek 的 API 密钥：

```
DEEPSEEK_API_KEY=sk-xxxxxxx
```

如未设置，将使用默认密钥（仅作测试用途）。

---

## 📌 TODO（可拓展方向）

* 多文档上传支持
* 文档分类标签管理
* 支持本地大语言模型（如 Qwen/ChatGLM）
* 向量库刷新优化为增量式

---

## 📮 联系方式

如有问题或建议，请联系项目作者或提交 Issue。

---

项目参考了 RAG 技术的典型实现，适合用于垂直行业文档问答、企业知识库接入、智能客服等应用场景。

```

---

需要我为你生成 requirements.txt 文件，或将所有文件组织成一个可下载的项目压缩包吗？
```
