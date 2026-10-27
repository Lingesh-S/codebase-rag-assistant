# embeddings/embedder.py

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

# Small, strong general-purpose retriever that runs on CPU through ONNX (no PyTorch).
# For larger codebases try "jinaai/jina-embeddings-v2-base-code" (code-specific, 768-d, ~640 MB).
DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings backed by fastembed, using the model's own passage/query prefixes."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 32):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly; far more precise than vector similarity for the final ordering."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, passages)]


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name)


@lru_cache(maxsize=2)
def get_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)
