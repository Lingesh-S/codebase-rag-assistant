# rag_pipeline/pipeline.py

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from embeddings.embedder import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL, get_embedding_model, get_reranker
from loaders.code_loader import load_codebase
from rag_pipeline.generator import (
    LLM,
    MAX_CONTEXT_CHARS,
    SYSTEM_PROMPT,
    build_messages,
    condense_question,
    is_follow_up,
)
from retriever.code_retriever import HybridRetriever
from utils.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from vectorstore.vector_store import CodeVectorStore, IndexStats, StoredChunk, collection_name_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RAGSettings:
    # None keeps the index in memory only.
    persist_dir: str | None = field(
        default_factory=lambda: os.getenv("RAG_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
    )
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rerank_model: str | None = DEFAULT_RERANK_MODEL  # loaded on first use; None disables reranking
    rerank_top_n: int = 10
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = 6
    candidate_k: int = 20
    retrieval_mode: str = "hybrid"
    max_context_chars: int = MAX_CONTEXT_CHARS


@dataclass
class Answer:
    question: str
    search_query: str
    sources: list[StoredChunk]
    stream: Iterator[str] | None = None  # None in retrieval-only mode (no LLM configured)


class RAGPipeline:
    """Index a codebase and answer questions about it with cited sources."""

    def __init__(
        self,
        codebase_path,
        settings: RAGSettings | None = None,
        collection_name: str | None = None,
        embedding=None,
        reranker=None,
    ):
        self.codebase_path = Path(codebase_path).resolve()
        self.settings = settings or RAGSettings()
        if embedding is None:
            embedding = get_embedding_model(self.settings.embedding_model)
        self.store = CodeVectorStore(
            embedding,
            collection_name or collection_name_for(str(self.codebase_path)),
            self.settings.persist_dir,
        )
        self.retriever = HybridRetriever(
            self.store, reranker, candidate_k=self.settings.candidate_k, rerank_top_n=self.settings.rerank_top_n
        )

    def build_knowledge_base(self, progress: Callable[[str, int, int], None] | None = None) -> IndexStats:
        """
        Loads, chunks and indexes the codebase; only new or changed files are re-embedded.

        progress(stage, done, total) is called as work advances, so a UI can show it.
        """
        report = progress or (lambda stage, done, total: None)
        report("Reading files", 0, 0)
        documents = load_codebase(self.codebase_path)
        report("Splitting code", len(documents), len(documents))
        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        stats = self.store.sync(chunks, progress=lambda done, total: report("Embedding", done, total))
        report("Building keyword index", stats.total_chunks, stats.total_chunks)
        self.retriever.refresh()
        return stats

    def retrieve(self, query: str, k: int | None = None, rerank: bool = False) -> list[StoredChunk]:
        if rerank and self.retriever.reranker is None and self.settings.rerank_model:
            self.retriever.reranker = get_reranker(self.settings.rerank_model)
        return self.retriever.retrieve(query, k or self.settings.top_k, self.settings.retrieval_mode, rerank)

    def stream_answer(
        self,
        question: str,
        llm: LLM | None = None,
        history: list[dict] | None = None,
        rewrite_followups: bool = True,
        k: int | None = None,
        rerank: bool = False,
        on_step: Callable[[str], None] | None = None,
    ) -> Answer:
        step = on_step or (lambda label: None)
        search_query = question
        if llm is not None and history and rewrite_followups and is_follow_up(question):
            step("Rewriting the follow-up question")
            search_query = condense_question(llm, question, history)
        step("Searching the codebase")
        chunks = self.retrieve(search_query, k, rerank)
        if llm is None:
            return Answer(question, search_query, chunks)
        messages, used = build_messages(question, chunks, history, self.settings.max_context_chars)
        step("Writing the answer")
        return Answer(question, search_query, used, llm.stream(SYSTEM_PROMPT, messages))

    def answer(self, question: str, llm: LLM, history: list[dict] | None = None, **kwargs) -> tuple[str, Answer]:
        result = self.stream_answer(question, llm, history, **kwargs)
        return "".join(result.stream or []), result
