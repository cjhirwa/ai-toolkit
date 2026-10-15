"""retrievekit — swappable retrieval strategies behind one ``retrieve(query, k, filters)`` call."""

from .bench import benchmark, evaluate, format_table, load_qrels
from .core import Doc, Filter, Hit, Retriever, load_docs, matches
from .fusion import HybridRetriever, rrf
from .index import Index
from .pipeline import Pipeline
from .rerank import CrossEncoderReranker, LLMReranker, Reranker
from .retrievers import BM25Retriever, ChromaRetriever, DenseRetriever, tokenize
from .transforms import HyDE, MultiQuery, QueryTransform

__all__ = [
    "Doc", "Hit", "Filter", "Retriever", "load_docs", "matches",
    "BM25Retriever", "DenseRetriever", "ChromaRetriever", "tokenize",
    "HybridRetriever", "rrf", "Pipeline", "Index",
    "Reranker", "LLMReranker", "CrossEncoderReranker", "QueryTransform", "MultiQuery", "HyDE",
    "benchmark", "evaluate", "format_table", "load_qrels",
]
