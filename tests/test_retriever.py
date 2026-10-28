import pytest

from retriever.code_retriever import reciprocal_rank_fusion, stem, tokenize_code
from tests.conftest import write


def test_tokenizer_splits_camel_and_snake_case_identifiers():
    tokens = set(tokenize_code("loadConfigFile(path) calls parse_json_body and HTTPServer"))

    assert set(tokenize_code("load config file")) <= tokens
    assert set(tokenize_code("parse json body")) <= tokens
    assert set(tokenize_code("http server")) <= tokens
    assert stem("parse_json_body") in tokens


def test_tokenizer_matches_natural_language_to_identifiers():
    question = set(tokenize_code("How are passwords hashed?"))
    code = set(tokenize_code("def hash_password(password): '''Hashes a password.'''"))

    assert question == {"password", "hash"}
    assert question <= code


def test_stemming_is_consistent_across_word_forms():
    assert stem("retries") == stem("retry")
    assert stem("parsing") == stem("parsed") == stem("parses") == stem("parse")
    assert stem("classes") == stem("class") == "class"


def test_tokenizer_drops_question_stopwords():
    assert tokenize_code("How does the cart work?") == ["cart", "work"]


def test_rrf_rewards_documents_found_by_both_retrievers():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d", "a"]], k=60)
    ranked = sorted(scores, key=scores.get, reverse=True)

    assert ranked[:2] == ["a", "c"]
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)


def test_bm25_matches_exact_identifiers(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()

    [top, *_] = rag.retriever.bm25_search("fetchWithRetry", k=3)

    assert top.metadata["source"] == "web/client.js"


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How are passwords hashed?", "app/auth.py"),
        ("add item to the shopping cart", "app/cart.py"),
        ("retry failed fetch requests", "web/client.js"),
    ],
)
def test_every_mode_finds_the_relevant_file(codebase, make_rag, mode, question, expected):
    rag = make_rag(codebase, retrieval_mode=mode)
    rag.build_knowledge_base()

    results = rag.retrieve(question, k=3)

    assert results[0].metadata["source"] == expected


class KeywordReranker:
    """Scores passages by whether they mention a keyword, to prove reranking reorders results."""

    def __init__(self, keyword):
        self.keyword = keyword

    def score(self, query, passages):
        return [1.0 if self.keyword in p else 0.0 for p in passages]


def test_reranker_reorders_candidates(codebase, make_rag):
    rag = make_rag(codebase, reranker=KeywordReranker("max_cart_items"))
    rag.build_knowledge_base()

    reranked = rag.retrieve("shopping cart items", k=3, rerank=True)
    plain = rag.retrieve("shopping cart items", k=3, rerank=False)

    assert reranked[0].metadata["source"] == "settings.json"
    assert plain[0].metadata["source"] == "app/cart.py"


def test_only_the_top_candidates_are_reranked(codebase, make_rag):
    rag = make_rag(codebase, reranker=KeywordReranker("max_cart_items"), rerank_top_n=1)
    rag.build_knowledge_base()

    plain = rag.retrieve("shopping cart items", k=10, rerank=False)
    reranked = rag.retrieve("shopping cart items", k=10, rerank=True)

    assert [c.id for c in reranked] == [c.id for c in plain]


def test_bm25_index_refreshes_after_reindexing(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()
    rag.retriever.bm25_search("anything", k=1)
    write(codebase, "app/payments.py", "def charge_credit_card(amount):\n    return amount\n")
    rag.build_knowledge_base()

    [top, *_] = rag.retriever.bm25_search("charge_credit_card", k=1)
    assert top.metadata["source"] == "app/payments.py"


def test_unknown_mode_is_rejected(codebase, make_rag):
    rag = make_rag(codebase)
    with pytest.raises(ValueError):
        rag.retriever.retrieve("x", mode="magic")
