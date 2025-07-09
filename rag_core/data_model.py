"""
定义数据结构：Document 和 QueryResult
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

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