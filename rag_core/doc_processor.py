"""
文档预处理模块：包括清洗、分句、切块等
"""
import re
from typing import List, Dict
from .data_model import Document

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
        return [
            Document(id=f"{source}_{i}", title=title, content=chunk, source=source,
                     metadata={**metadata, 'chunk_index': i, 'total_chunks': len(chunks)})
            for i, chunk in enumerate(chunks)
        ]
