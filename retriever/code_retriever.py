# retriever/code_retriever.py

import re
from collections import defaultdict
from dataclasses import replace

from rank_bm25 import BM25Okapi

from vectorstore.vector_store import CodeVectorStore, StoredChunk

RETRIEVAL_MODES = ("hybrid", "dense", "bm25")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "show",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "file",
    "lines",
    "symbol",
}

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_SUBWORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def stem(word: str) -> str:
    """Tiny suffix stripper so "hashed", "hashes" and "hash" (or "passwords"/"password") match."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("es"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize_code(text: str) -> list[str]:
    """
    Lexical tokens for BM25 that understand identifiers.

    "loadConfigFile" and "load_config_file" both produce the full identifier plus
    "load", "config" and "file", so natural language questions match code names.
    """
    tokens = []
    for word in _WORD.findall(text):
        lower = word.lower()
        if lower not in STOPWORDS:
            tokens.append(stem(lower))
        parts = _SUBWORD.findall(word)
        if len(parts) > 1:
            tokens.extend(stem(p.lower()) for p in parts if p.lower() not in STOPWORDS)
    return tokens


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Merges ranked id lists; each list contributes 1 / (k + rank) for every id it contains."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


class HybridRetriever:
    """
    Dense vector search + BM25 keyword search, fused with Reciprocal Rank Fusion,
    then optionally re-ordered by a cross-encoder reranker.
    """

    def __init__(
        self, store: CodeVectorStore, reranker=None, candidate_k: int = 20, rrf_k: int = 60, rerank_top_n: int = 10
    ):
        self.store = store
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        # Cross-encoders cost ~0.3 s per passage on a laptop CPU, so only the head of the list is reranked.
        self.rerank_top_n = rerank_top_n
        self._chunks: list[StoredChunk] = []
        self._bm25: BM25Okapi | None = None

    def refresh(self) -> None:
        """Rebuilds the in-memory BM25 index from the vector store."""
        self._chunks = self.store.all_chunks()
        corpus = [tokenize_code(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None

    def bm25_search(self, query: str, k: int) -> list[StoredChunk]:
        if self._bm25 is None or len(self._chunks) != self.store.count():
            self.refresh()
        tokens = tokenize_code(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [replace(self._chunks[i], score=float(scores[i])) for i in order if scores[i] > 0]

    def retrieve(self, query: str, k: int = 6, mode: str = "hybrid", rerank: bool = True) -> list[StoredChunk]:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}")

        if mode == "dense":
            candidates = self.store.dense_search(query, self.candidate_k)
        elif mode == "bm25":
            candidates = self.bm25_search(query, self.candidate_k)
        else:
            dense = self.store.dense_search(query, self.candidate_k)
            sparse = self.bm25_search(query, self.candidate_k)
            fused = reciprocal_rank_fusion([[c.id for c in dense], [c.id for c in sparse]], self.rrf_k)
            by_id = {c.id: c for c in dense + sparse}
            candidates = [
                replace(by_id[chunk_id], score=score)
                for chunk_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)
            ]

        if rerank and self.reranker is not None and candidates:
            head, tail = candidates[: self.rerank_top_n], candidates[self.rerank_top_n :]
            scores = self.reranker.score(query, [c.text for c in head])
            ranked = sorted(zip(head, scores, strict=True), key=lambda pair: pair[1], reverse=True)
            candidates = [replace(c, score=s) for c, s in ranked] + tail
        return candidates[:k]
