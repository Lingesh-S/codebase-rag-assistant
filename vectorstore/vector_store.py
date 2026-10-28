# vectorstore/vector_store.py

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

BATCH_SIZE = 128


@dataclass
class IndexStats:
    files_added: int = 0
    files_updated: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    chunks_added: int = 0
    total_files: int = 0
    total_chunks: int = 0


@dataclass
class StoredChunk:
    id: str
    text: str
    metadata: dict
    score: float = 0.0


def collection_name_for(identity: str) -> str:
    """Stable, Chroma-safe collection name for a codebase (e.g. its resolved folder path)."""
    return "code-" + hashlib.sha1(identity.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def batched(items: list, size: int = BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class CodeVectorStore:
    """
    Chroma collection holding the chunks of one codebase.

    sync() makes the collection mirror a fresh set of chunks: files whose content
    hash is unchanged are skipped, changed files are re-embedded and deleted files
    are removed, so rebuilding never duplicates vectors.
    """

    def __init__(self, embedding: Embeddings, collection_name: str, persist_dir: str | None = "chroma_db"):
        self.embedding = embedding
        self.embedding_model = getattr(embedding, "model_name", type(embedding).__name__)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        self.collection = self._open_collection()

    def _open_collection(self):
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
            embedding_function=None,
        )
        if (collection.metadata or {}).get("embedding_model") != self.embedding_model:
            # Vectors from another model live in a different space, so start over.
            self.client.delete_collection(self.collection_name)
            return self._open_collection()
        return collection

    def sync(self, chunks: list[Document]) -> IndexStats:
        existing = self.collection.get(include=["metadatas"])
        indexed_hash: dict[str, str] = {}
        ids_by_source: dict[str, list[str]] = defaultdict(list)
        for chunk_id, meta in zip(existing["ids"], existing["metadatas"], strict=True):
            indexed_hash[meta["source"]] = meta.get("file_hash", "")
            ids_by_source[meta["source"]].append(chunk_id)

        current_hash = {c.metadata["source"]: c.metadata["file_hash"] for c in chunks}
        stats = IndexStats(total_files=len(current_hash))
        stale_ids: list[str] = []
        for source, file_hash in indexed_hash.items():
            if source not in current_hash:
                stats.files_removed += 1
                stale_ids += ids_by_source[source]
            elif current_hash[source] != file_hash:
                stats.files_updated += 1
                stale_ids += ids_by_source[source]
            else:
                stats.files_unchanged += 1
        stats.files_added = len(current_hash.keys() - indexed_hash.keys())

        for batch in batched(stale_ids):
            self.collection.delete(ids=batch)

        new_chunks = [c for c in chunks if indexed_hash.get(c.metadata["source"]) != c.metadata["file_hash"]]
        for batch in batched(new_chunks):
            texts = [c.page_content for c in batch]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=texts,
                metadatas=[c.metadata for c in batch],
                embeddings=self.embedding.embed_documents(texts),
            )
        stats.chunks_added = len(new_chunks)
        stats.total_chunks = self.collection.count()
        return stats

    def dense_search(self, query: str, k: int = 20) -> list[StoredChunk]:
        total = self.collection.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedding.embed_query(query)],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            StoredChunk(id=i, text=t, metadata=m, score=1.0 - d)
            for i, t, m, d in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]

    def all_chunks(self) -> list[StoredChunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            StoredChunk(id=i, text=t, metadata=m)
            for i, t, m in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self._open_collection()
