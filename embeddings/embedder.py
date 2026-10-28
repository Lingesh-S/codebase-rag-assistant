# embeddings/embedder.py

import os
import threading
from contextlib import suppress
from functools import lru_cache

from langchain_core.embeddings import Embeddings

# Small, strong general-purpose retriever that runs through ONNX (no PyTorch).
# For larger codebases try "jinaai/jina-embeddings-v2-base-code" (code-specific, 768-d, ~640 MB).
DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
# "auto" uses an NVIDIA GPU when the GPU build of ONNX Runtime is installed, else the CPU; "cpu" forces the CPU.
EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "auto")
CPU_BATCH_SIZE = 32
GPU_BATCH_SIZE = 64  # 32 to 128 run at the same ~500 chunks/s on an RTX 4050; 64 keeps memory low


def onnx_providers(device: str = EMBEDDING_DEVICE) -> list[str]:
    """ONNX Runtime execution providers to try, best first; the CPU is always the fallback."""
    import onnxruntime

    if device != "cpu" and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        # Lets the GPU build find CUDA and cuDNN installed as nvidia-* pip packages.
        with suppress(Exception):
            onnxruntime.preload_dlls()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def active_device(model) -> str:
    """'cuda' only if ONNX Runtime really started the model on the GPU; it falls back to the CPU silently."""
    with suppress(Exception):
        if "CUDAExecutionProvider" in model.model.model.get_providers():
            return "cuda"
    return "cpu"


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings backed by fastembed, using the model's own passage/query prefixes."""

    def __init__(
        self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE, batch_size: int | None = None
    ):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, providers=onnx_providers(device))
        self.device = active_device(self._model)
        self.batch_size = batch_size or (GPU_BATCH_SIZE if self.device == "cuda" else CPU_BATCH_SIZE)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly; far more precise than vector similarity for the final ordering."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: str = EMBEDDING_DEVICE):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, providers=onnx_providers(device))
        self.device = active_device(self._model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, passages)]


# A warm-up thread and an indexing job may ask for a model at the same moment; they must share one instance.
_model_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load_embedding_model(model_name: str) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name)


@lru_cache(maxsize=2)
def _load_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> FastEmbedEmbeddings:
    with _model_lock:
        return _load_embedding_model(model_name)


def get_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoderReranker:
    with _model_lock:
        return _load_reranker(model_name)
