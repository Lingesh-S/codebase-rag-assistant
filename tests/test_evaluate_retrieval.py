import pytest

from evaluation import evaluate_retrieval
from tests.conftest import FakeEmbeddings


class KeepOrderReranker:
    def score(self, query, passages):
        return [-i for i in range(len(passages))]


@pytest.mark.parametrize("store", ["chroma", "qdrant"])
def test_benchmark_runs_on_either_store(monkeypatch, capsys, qdrant_memory, store):
    monkeypatch.setattr(evaluate_retrieval, "get_embedding_model", lambda name: FakeEmbeddings())
    monkeypatch.setattr(evaluate_retrieval, "get_reranker", lambda name: KeepOrderReranker())

    evaluate_retrieval.main(["--store", store, "--qdrant-url", ":memory:"])

    out = capsys.readouterr().out
    assert f"store: {store}" in out
    assert "hybrid + rerank" in out
    assert [c.name for c in qdrant_memory.get_collections().collections] == []
