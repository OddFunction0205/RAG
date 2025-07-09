import os
import pickle
import faiss
import numpy as np
import importlib
from .data_model import Document


class VectorStore:
    def __init__(self, embedding_dim: int, index_path="./vector_database/vector.index", docs_path="./vector_database/documents.pkl"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.docs_path = docs_path
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)

            # 修复 pickle 加载失败的问题
            def custom_load(file):
                globals()['Document'] = importlib.import_module('rag_core.data_model').Document
                return pickle.load(file)

            with open(docs_path, "rb") as f:
                self.documents = custom_load(f)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.documents = []

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_documents(self, docs: list[Document], embeddings: np.ndarray):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for i, doc in enumerate(docs):
            doc.embedding = embeddings[i]
        self.documents.extend(docs)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k=5):
        if not self.documents:
            return []
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [(self.documents[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if 0 <= i < len(self.documents)]

    def list_documents(self, max_chars=30) -> list[str]:
        return [f"[{i}] {doc.title} - {doc.content[:max_chars]}..." for i, doc in enumerate(self.documents)]

    def search_by_keyword(self, keyword: str, max_results=10) -> list[str]:
        result = [f"[{i}] {doc.title} - {doc.content[:30]}..." for i, doc in enumerate(self.documents)
                  if keyword in doc.content or keyword in doc.title]
        return result[:max_results] if result else ["未找到匹配结果。"]

    def delete_document_by_index(self, index: int) -> str:
        if 0 <= index < len(self.documents):
            # 获取要删除的片段的来源文件名
            deleted_doc = self.documents[index]
            source_file = deleted_doc.source  # 存储的是原始上传文件名（可能是 PDF / DOCX / TXT）

            # 删除该文档片段
            del self.documents[index]

            # 检查是否还有其他片段来自同一个原始文档
            remaining_sources = [doc.source for doc in self.documents]
            if source_file not in remaining_sources:
                # 删除原始上传的文档文件（可能是 PDF / DOCX / TXT）
                sub_dirs = ["pdf", "docx", "txt"]
                deleted = False
                for sub_dir in sub_dirs:
                    raw_path = os.path.join("raw_data", sub_dir, source_file)
                    if os.path.exists(raw_path):
                        os.remove(raw_path)
                        msg = f"并删除了原始文档文件：{os.path.join(sub_dir, source_file)}"
                        deleted = True
                        break
                if not deleted:
                    msg = f"（原始文档文件 {source_file} 不存在或已删除）"
            else:
                msg = f"原始文档文件 {source_file} 仍有其他片段存在，未删除。"

            # 重建向量索引
            embeddings = np.array([doc.embedding for doc in self.documents if doc.embedding is not None])
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if len(embeddings) > 0:
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)

            self.save()
            return f"✅ 已删除第 {index} 个文档片段。{msg}"
        return f"❌ 无效索引：{index}"
