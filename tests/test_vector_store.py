from tests.conftest import FakeEmbeddings, write
from vectorstore.vector_store import CodeVectorStore, collection_name_for


def sources_in(rag) -> set[str]:
    return {c.metadata["source"] for c in rag.store.all_chunks()}


def test_rebuilding_does_not_duplicate_chunks(codebase, make_rag):
    rag = make_rag(codebase)
    first = rag.build_knowledge_base()
    second = rag.build_knowledge_base()

    assert first.files_added == 5
    assert first.total_chunks > 0
    assert second.total_chunks == first.total_chunks
    assert second.chunks_added == 0
    assert second.files_unchanged == 5


def test_index_persists_and_is_reused_across_instances(codebase, make_rag):
    first = make_rag(codebase).build_knowledge_base()
    reopened = make_rag(codebase)

    assert reopened.store.count() == first.total_chunks
    assert reopened.build_knowledge_base().chunks_added == 0


def test_changed_file_is_reindexed_and_old_chunks_removed(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()
    write(codebase, "app/auth.py", "def verify_token(token):\n    return token == 'ok'\n")

    stats = rag.build_knowledge_base()

    assert (stats.files_updated, stats.files_unchanged) == (1, 4)
    texts = [c.text for c in rag.store.all_chunks() if c.metadata["source"] == "app/auth.py"]
    assert texts and all("verify_token" in t and "hash_password" not in t for t in texts)


def test_deleted_file_is_removed_from_index(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()
    (codebase / "web/client.js").unlink()

    stats = rag.build_knowledge_base()

    assert stats.files_removed == 1
    assert "web/client.js" not in sources_in(rag)


def test_switching_embedding_model_rebuilds_collection(tmp_path, codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()

    other = CodeVectorStore(FakeEmbeddings(dim=128, model_name="other-model"), "test-code", str(tmp_path / "chroma"))

    assert other.count() == 0


def test_collection_names_are_stable_and_chroma_safe():
    name = collection_name_for("C:/Users/me/My Project")
    assert name == collection_name_for("C:/Users/me/My Project")
    assert name.startswith("code-") and name.replace("-", "").isalnum()


def test_sync_reports_embedding_progress_per_batch(codebase, make_rag, monkeypatch):
    from vectorstore import vector_store

    monkeypatch.setattr(vector_store, "BATCH_SIZE", 2)
    rag = make_rag(codebase)
    calls = []

    stats = rag.store.sync(rag_chunks(codebase), progress=lambda done, total: calls.append((done, total)))

    assert calls[-1] == (stats.chunks_added, stats.chunks_added)
    assert [done for done, _ in calls] == sorted(done for done, _ in calls)
    assert len(calls) > 1


def test_unchanged_sync_reports_no_embedding_work(codebase, make_rag):
    rag = make_rag(codebase)
    rag.store.sync(rag_chunks(codebase))
    calls = []

    rag.store.sync(rag_chunks(codebase), progress=lambda done, total: calls.append((done, total)))

    assert calls == []


def rag_chunks(root):
    from loaders.code_loader import load_codebase
    from utils.chunker import chunk_documents

    return chunk_documents(load_codebase(root))
