"""
向量编码器模块：使用 SentenceTransformer 构建文本向量
"""
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_texts([query])[0]