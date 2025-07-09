# 🤖 基于 RAG 的中文行业知识问答系统

本项目构建了一个**基于检索增强生成（RAG）框架**的中文问答系统。用户可上传行业文档（如 PDF），系统将文档切分、向量化并构建知识库，结合大语言模型（DeepSeek-Chat）进行精准问答。同时提供简洁的 HTML/CSS/JS 前端界面，便于用户交互体验。


## 🏗️ 项目结构

```
.
├── app.py                    # FastAPI 主程序，提供 API 服务并挂载前端
├── rag_core/                 # RAG 核心功能模块
│   ├── __init__.py           # 包初始化文件
│   ├── data_model.py         # 数据模型定义（文档结构、数据类等）
│   ├── doc_processor.py      # 文档解析与预处理（清洗、切分等）
│   ├── embedder.py           # 文本嵌入向量生成（调用 Sentence-Transformers 等）
│   ├── generator.py          # 大模型接口封装（调用 DeepSeek API 生成回答）
│   ├── vector_store.py       # FAISS 向量数据库管理（存储、检索、更新向量）
│   └── __pycache__/          # Python 编译缓存文件
├── old/                      # 历史版本代码（旧脚本备份）
│   ├── main.py
│   └── temp.py
├── raw_data/                 # 原始文档存储目录
│   ├── docx/                 # DOCX 格式文档
│   ├── pdf/                  # PDF 格式文档
│   └── txt/                  # TXT 格式文档
├── vector_database/          # 向量数据库文件
│   ├── documents.pkl         # 文档元数据序列化文件
│   └── vector.index          # FAISS 向量索引文件
├── __pycache__/              # 主程序编译缓存
│   └── app.cpython-312.pyc
├── run_dev.py                # 开发是用来监听的watch dog
├── style.css
├── README.md                 # 项目说明文档
├── result.md                 # 结果记录文档
├── 第1天开发日志.md          # 开发日志（第一天）
├── 第2天开发日志.md          # 开发日志（第二天）
└── 第3天开发日志.md          # 开发日志（第三天）
```


## 🚀 快速开始

### 1️⃣ 安装依赖

建议使用 `conda` 或 `venv` 创建虚拟环境后安装：

```bash
pip install -r requirements.txt
```


### 2️⃣ 启动服务

```bash
python ./app.py
```



## 📦 功能模块说明

### ✅ 文档上传与预处理

- 支持 PDF、DOCX、TXT 文件上传
- 自动清洗文本、分句、切块
- 构造文档段落并生成对应向量

### ✅ 问答功能（RAG Pipeline）

- 用户输入问题后：
  1. 系统向量化问题语义
  2. 通过 FAISS 检索最相关的文档片段
  3. 利用 DeepSeek-Chat 大模型生成答案
- 支持来源追溯（返回回答所用片段）

### ✅ 向量数据库管理

- 查看当前知识库片段（编号+摘要）
- 关键词搜索（全文匹配）
- 支持单个或批量删除片段（如：3,5-7,10）


## 🔐 API 密钥配置（DeepSeek）

请在根目录下创建 `.env` 文件或设置环境变量：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

如未设置，将使用系统默认测试密钥（不推荐正式部署使用）。


## 🧱 可扩展方向（TODO）

- ✅ 多文档并行上传
- ✅ 批量片段删除
- ⏳ 支持文档分类 / 标签系统
- ⏳ 向量库更新改为增量式
- ⏳ 支持本地语言模型（如 Qwen、ChatGLM）
- ⏳ 增加回答评分 / 用户反馈接口


## 📮 联系方式

如有问题或建议，欢迎提交 Issue 或联系开发者：

- 作者：OddFunction0205
- 邮箱：oddfunction0205@163.com


本项目适用于企业知识库问答系统、智能客服、垂直领域搜索助手等应用场景，欢迎继续扩展与定制。