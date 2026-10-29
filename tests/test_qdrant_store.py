import pytest

from rag_pipeline.pipeline import RAGSettings
from retriever.code_retriever import tokenize_code
from tests.conftest import FakeEmbeddings, write
from tests.test_vector_store import rag_chunks
from vectorstore.qdrant_store import QdrantStore, point_id, sparse_vector
from vectorstore.vector_store import CodeVectorStore


@pytest.fixture
def make_qdrant_rag(make_rag, qdrant_memory):
    def factory(root, **settings):
        return make_rag(root, vector_store="qdrant", qdrant_url=":memory:", **settings)

    return factory


def sources_in(store) -> set[str]:
    return {c.metadata["source"] for c in store.all_chunks()}


def test_pipeline_picks_the_store_from_settings(codebase, make_rag, make_qdrant_rag):
    assert isinstance(make_rag(codebase).store, CodeVectorStore)
    assert isinstance(make_qdrant_rag(codebase).store, QdrantStore)


def test_store_defaults_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_STORE", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    settings = RAGSettings()

    assert (settings.vector_store, settings.qdrant_url) == ("qdrant", "http://qdrant:6333")


def test_unknown_store_names_are_rejected(codebase, make_rag):
    with pytest.raises(ValueError, match="RAG_VECTOR_STORE"):
        make_rag(codebase, vector_store="pinecone")


def test_rebuilding_does_not_duplicate_points(codebase, make_qdrant_rag):
    rag = make_qdrant_rag(codebase)
    first = rag.build_knowledge_base()
    second = rag.build_knowledge_base()

    assert first.files_added == 5
    assert first.total_chunks > 0
    assert (second.total_chunks, second.chunks_added, second.files_unchanged) == (first.total_chunks, 0, 5)


def test_index_is_shared_by_every_store_on_the_same_server(codebase, make_qdrant_rag):
    first = make_qdrant_rag(codebase).build_knowledge_base()
    reopened = make_qdrant_rag(codebase)

    assert reopened.store.count() == first.total_chunks
    assert reopened.build_knowledge_base().chunks_added == 0


def test_changed_and_deleted_files_are_diffed_per_file(codebase, make_qdrant_rag):
    rag = make_qdrant_rag(codebase)
    rag.build_knowledge_base()
    write(codebase, "app/auth.py", "def verify_token(token):\n    return token == 'ok'\n")
    (codebase / "web/client.js").unlink()

    stats = rag.build_knowledge_base()

    assert (stats.files_updated, stats.files_removed, stats.files_unchanged) == (1, 1, 3)
    texts = [c.text for c in rag.store.all_chunks() if c.metadata["source"] == "app/auth.py"]
    assert texts and all("verify_token" in t and "hash_password" not in t for t in texts)
    assert "web/client.js" not in sources_in(rag.store)


def test_points_keep_the_chunk_ids_and_metadata(codebase, make_qdrant_rag):
    rag = make_qdrant_rag(codebase)
    chunks = rag_chunks(codebase)
    rag.store.sync(chunks)

    stored = {c.id: c for c in rag.store.all_chunks()}

    assert set(stored) == {c.id for c in chunks}
    original = chunks[0]
    assert stored[original.id].text == original.page_content
    assert stored[original.id].metadata == original.metadata


def test_point_ids_are_deterministic_uuids():
    chunk_id = "0123456789abcdef0123456789abcdef01234567"

    assert point_id(chunk_id) == point_id(chunk_id)
    assert point_id(chunk_id) != point_id(chunk_id[::-1])


def test_sparse_vectors_hash_the_code_tokens():
    vector = sparse_vector(tokenize_code("hash_password hashes the password"))

    assert len(vector.indices) == len(set(vector.indices)) == len(vector.values)
    assert all(0 <= i < 2**31 for i in vector.indices)
    assert sparse_vector(tokenize_code("password")).indices[0] in vector.indices
    assert sparse_vector([]).indices == []


def test_switching_embedding_model_rebuilds_the_collection(codebase, make_qdrant_rag):
    make_qdrant_rag(codebase).build_knowledge_base()

    other = QdrantStore(FakeEmbeddings(dim=128, model_name="other-model"), "test-code", ":memory:")

    assert other.count() == 0


def test_reset_empties_the_collection(codebase, make_qdrant_rag):
    rag = make_qdrant_rag(codebase)
    rag.build_knowledge_base()

    rag.store.reset()

    assert rag.store.count() == 0
    assert rag.store.dense_search("cart") == []


def test_sync_reports_embedding_progress_per_batch(codebase, make_qdrant_rag, monkeypatch):
    from vectorstore import vector_store

    monkeypatch.setattr(vector_store, "BATCH_SIZE", 2)
    rag = make_qdrant_rag(codebase)
    calls = []

    stats = rag.store.sync(rag_chunks(codebase), progress=lambda done, total: calls.append((done, total)))

    assert calls[-1] == (stats.chunks_added, stats.chunks_added)
    assert len(calls) > 1


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How are passwords hashed?", "app/auth.py"),
        ("add item to the shopping cart", "app/cart.py"),
        ("fetchWithRetry", "web/client.js"),
    ],
)
@pytest.mark.parametrize("search", ["dense_search", "hybrid_search"])
def test_searches_find_the_relevant_file(codebase, make_qdrant_rag, search, question, expected):
    rag = make_qdrant_rag(codebase)
    rag.build_knowledge_base()

    [top, *_] = getattr(rag.store, search)(question, k=3)

    assert top.metadata["source"] == expected
    assert top.score > 0


def test_hybrid_search_filters_by_path_prefix_and_language(codebase, make_qdrant_rag):
    rag = make_qdrant_rag(codebase)
    rag.build_knowledge_base()

    in_web = rag.store.hybrid_search("password cart retry", k=10, path_prefix="web")
    python = rag.store.hybrid_search("password cart retry", k=10, language="python")
    prefix_is_a_folder = rag.store.hybrid_search("password", k=10, path_prefix="ap")

    assert {c.metadata["source"] for c in in_web} == {"web/client.js"}
    assert {c.metadata["source"] for c in python} == {"app/auth.py", "app/cart.py"}
    assert prefix_is_a_folder == []


def test_retriever_uses_the_native_hybrid_query(codebase, make_qdrant_rag, monkeypatch):
    rag = make_qdrant_rag(codebase)
    rag.build_knowledge_base()
    monkeypatch.setattr(rag.retriever, "bm25_search", lambda *a, **kw: pytest.fail("BM25 is done by Qdrant"))

    [top, *_] = rag.retrieve("How are passwords hashed?")

    assert top.metadata["source"] == "app/auth.py"


@pytest.mark.parametrize("mode", ["dense", "bm25"])
def test_other_retrieval_modes_still_work_on_qdrant(codebase, make_qdrant_rag, mode):
    rag = make_qdrant_rag(codebase, retrieval_mode=mode)
    rag.build_knowledge_base()

    [top, *_] = rag.retrieve("add item to the shopping cart")

    assert top.metadata["source"] == "app/cart.py"


@pytest.mark.parametrize("store", ["chroma", "qdrant"])
def test_retriever_filters_work_on_both_stores(codebase, make_rag, qdrant_memory, store):
    rag = make_rag(codebase, vector_store=store, qdrant_url=":memory:")
    rag.build_knowledge_base()

    results = rag.retriever.retrieve("password cart retry", k=10, path_prefix="app", language="python")

    assert results
    assert {c.metadata["source"] for c in results} <= {"app/auth.py", "app/cart.py"}
