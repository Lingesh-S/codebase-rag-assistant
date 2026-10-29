# vectorstore/qdrant_store.py
#
# Qdrant collection for one codebase. Unlike Chroma it takes writes from several processes, so the
# index worker and the API can share it, and it runs hybrid search (dense + keyword) in one query.

import functools
import hashlib
import uuid
from collections import Counter
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from retriever.code_retriever import tokenize_code
from vectorstore.vector_store import IndexStats, StoredChunk, batched, plan_sync

DENSE, SPARSE = "dense", "sparse"
SPARSE_SPACE = 2**31
# BM25 term-frequency saturation; Qdrant's IDF modifier supplies the other half of the score.
BM25_K1, BM25_B, BM25_AVG_TOKENS = 1.2, 0.75, 100


@functools.cache
def qdrant_client(url: str) -> QdrantClient:
    """One client per server; ":memory:" is a process-wide in-memory instance for tests and offline runs."""
    return QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url)


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be integers or UUIDs, so the chunk's SHA-1 id is folded into a UUID."""
    return str(uuid.UUID(hashlib.md5(chunk_id.encode(), usedforsecurity=False).hexdigest()))


def token_index(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big") % SPARSE_SPACE


def sparse_vector(tokens: list[str], query: bool = False) -> models.SparseVector:
    counts = Counter(token_index(t) for t in tokens)
    if query:
        weights = dict.fromkeys(counts, 1.0)
    else:
        norm = BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / BM25_AVG_TOKENS)
        weights = {i: tf * (BM25_K1 + 1) / (tf + norm) for i, tf in counts.items()}
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def path_prefixes(source: str) -> list[str]:
    """Every folder above a file plus the file itself, so a keyword match acts as a path-prefix filter."""
    parts = source.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


class QdrantStore:
    """
    Qdrant collection holding the chunks of one codebase, with the same interface as CodeVectorStore.

    Each point carries a dense embedding and a sparse keyword vector built from tokenize_code, so
    hybrid_search() fuses both rankings with RRF inside Qdrant.
    """

    def __init__(self, embedding: Embeddings, collection_name: str, url: str = "http://127.0.0.1:6333"):
        self.embedding = embedding
        self.embedding_model = getattr(embedding, "model_name", type(embedding).__name__)
        self.collection_name = collection_name
        self.client = qdrant_client(url)
        self.local = url == ":memory:"
        self._open_collection()

    def _open_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            if (info.config.metadata or {}).get("embedding_model") == self.embedding_model:
                return
            # Vectors from another model live in a different space, so start over.
            self.client.delete_collection(self.collection_name)
        size = len(self.embedding.embed_query("dimension probe"))
        self.client.create_collection(
            self.collection_name,
            vectors_config={DENSE: models.VectorParams(size=size, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            metadata={"embedding_model": self.embedding_model},
        )
        if self.local:
            return  # the in-memory engine ignores payload indexes
        for field in ("source", "file_hash", "language", "path_prefixes"):
            self.client.create_payload_index(self.collection_name, field, models.PayloadSchemaType.KEYWORD)

    def _scroll(self, payload) -> list[models.Record]:
        records, offset = [], None
        while True:
            page, offset = self.client.scroll(
                self.collection_name, limit=1000, offset=offset, with_payload=payload, with_vectors=False
            )
            records += page
            if offset is None:
                return records

    def sync(self, chunks: list[Document], progress: Callable[[int, int], None] | None = None) -> IndexStats:
        existing = self._scroll(["chunk_id", "source", "file_hash"])
        indexed = [(r.payload["chunk_id"], r.payload["source"], r.payload.get("file_hash", "")) for r in existing]
        stats, stale_ids, new_chunks = plan_sync(indexed, chunks)

        for batch in batched(stale_ids):
            self.client.delete(self.collection_name, models.PointIdsList(points=[point_id(i) for i in batch]))

        done = 0
        for batch in batched(new_chunks):
            texts = [c.page_content for c in batch]
            points = [
                models.PointStruct(
                    id=point_id(chunk.id),
                    vector={DENSE: dense, SPARSE: sparse_vector(tokenize_code(chunk.page_content))},
                    payload={
                        "chunk_id": chunk.id,
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                        "source": chunk.metadata["source"],
                        "file_hash": chunk.metadata["file_hash"],
                        "language": chunk.metadata.get("language", "text"),
                        "path_prefixes": path_prefixes(chunk.metadata["source"]),
                    },
                )
                for chunk, dense in zip(batch, self.embedding.embed_documents(texts), strict=True)
            ]
            self.client.upsert(self.collection_name, points)
            done += len(batch)
            if progress:
                progress(done, len(new_chunks))
        stats.chunks_added = len(new_chunks)
        stats.total_chunks = self.count()
        return stats

    @staticmethod
    def _filter(path_prefix: str | None, language: str | None) -> models.Filter | None:
        conditions = []
        if path_prefix:
            prefix = path_prefix.replace("\\", "/").strip("/")
            conditions.append(models.FieldCondition(key="path_prefixes", match=models.MatchValue(value=prefix)))
        if language:
            conditions.append(models.FieldCondition(key="language", match=models.MatchValue(value=language)))
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _chunks(points: list[models.ScoredPoint]) -> list[StoredChunk]:
        return [StoredChunk(p.payload["chunk_id"], p.payload["text"], p.payload["metadata"], p.score) for p in points]

    def dense_search(
        self, query: str, k: int = 20, path_prefix: str | None = None, language: str | None = None
    ) -> list[StoredChunk]:
        result = self.client.query_points(
            self.collection_name,
            query=self.embedding.embed_query(query),
            using=DENSE,
            query_filter=self._filter(path_prefix, language),
            limit=k,
        )
        return self._chunks(result.points)

    def hybrid_search(
        self,
        query: str,
        k: int = 20,
        candidate_k: int = 20,
        path_prefix: str | None = None,
        language: str | None = None,
    ) -> list[StoredChunk]:
        """Dense and keyword candidates fused with Reciprocal Rank Fusion, all in one Qdrant query."""
        where = self._filter(path_prefix, language)
        prefetch = [
            models.Prefetch(query=self.embedding.embed_query(query), using=DENSE, filter=where, limit=candidate_k)
        ]
        tokens = tokenize_code(query)
        if tokens:
            sparse = sparse_vector(tokens, query=True)
            prefetch.append(models.Prefetch(query=sparse, using=SPARSE, filter=where, limit=candidate_k))
        result = self.client.query_points(
            self.collection_name, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k
        )
        return self._chunks(result.points)

    def all_chunks(self) -> list[StoredChunk]:
        return [
            StoredChunk(r.payload["chunk_id"], r.payload["text"], r.payload["metadata"])
            for r in self._scroll(["chunk_id", "text", "metadata"])
        ]

    def count(self) -> int:
        return self.client.count(self.collection_name, exact=True).count

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._open_collection()
