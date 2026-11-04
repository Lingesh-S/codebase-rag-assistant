import pytest

from api.client import ApiError, RagClient, parse_sse
from rag_pipeline.pipeline import Sources, Step, Token
from tests.conftest import FakeLLM


def indexed(api: RagClient, folder) -> str:
    repo = api.add_repo(path=str(folder))
    job = api.wait_for_job(api.index(repo["id"])["id"])
    assert job["status"] == "done", job
    return repo["id"]


def test_parse_sse_reads_events_and_skips_comments():
    lines = [": ping", "event: step", 'data: {"label": "Searching"}', "", "event: token", 'data: {"text": "a"}', ""]

    assert list(parse_sse(iter(lines))) == [("step", {"label": "Searching"}), ("token", {"text": "a"})]


def test_parse_sse_joins_multi_line_data_and_flushes_the_last_event():
    lines = ["event: done", 'data: {"answer":', 'data: "ok"}']

    assert list(parse_sse(iter(lines))) == [("done", {"answer": "ok"})]


def test_health_and_repo_registration(make_api_client, codebase):
    api = make_api_client()

    assert api.health()["status"] == "ok"
    repo = api.add_repo(path=str(codebase))
    assert repo["status"] == "not_indexed"
    assert api.add_repo(path=str(codebase))["id"] == repo["id"]
    assert [r["id"] for r in api.repos()] == [repo["id"]]


def test_waiting_for_a_job_reports_progress_until_it_is_done(make_api_client, codebase):
    api = make_api_client()
    repo = api.add_repo(path=str(codebase))
    seen = []

    job = api.wait_for_job(api.index(repo["id"])["id"], on_progress=seen.append)

    assert job["status"] == "done"
    assert job["stats"]["total_files"] == 5
    assert seen and seen[-1] == job
    assert api.repo(repo["id"])["status"] == "ready"


def test_search_returns_chunks_like_the_pipeline(make_api_client, codebase, make_rag):
    api = make_api_client()
    repo_id = indexed(api, codebase)
    rag = make_rag(codebase)
    rag.build_knowledge_base()

    chunks = api.search(repo_id, "How are passwords hashed?", k=3)

    expected = rag.retrieve("How are passwords hashed?", 3)
    assert [c.metadata["source"] for c in chunks] == [c.metadata["source"] for c in expected]
    assert chunks[0].metadata["source"] == "app/auth.py"
    assert "def hash_password" in chunks[0].text
    assert chunks[0].metadata["header_lines"] == 0


def test_chat_events_are_the_pipelines_event_objects(make_api_client, codebase):
    api = make_api_client(llm=FakeLLM("It uses PBKDF2 [1]."))
    repo_id = indexed(api, codebase)

    events = list(api.chat_events(repo_id, "How are passwords hashed?", provider="ollama", mode="oneshot"))

    assert isinstance(events[0], Step)
    sources = next(e for e in events if isinstance(e, Sources))
    assert sources.search_query == "How are passwords hashed?"
    assert sources.chunks[0].metadata["source"] == "app/auth.py"
    assert "".join(e.text for e in events if isinstance(e, Token)) == "It uses PBKDF2 [1]."
    assert api.last_done["provider"] == "fake"


def test_ask_streams_the_answer_after_the_sources(make_api_client, codebase):
    api = make_api_client(llm=FakeLLM("It uses PBKDF2 [1]."))
    repo_id = indexed(api, codebase)
    steps = []

    answer = api.ask(repo_id, "How are passwords hashed?", provider="ollama", mode="oneshot", on_step=steps.append)

    assert answer.sources[0].metadata["source"] == "app/auth.py"
    assert "".join(answer.stream) == "It uses PBKDF2 [1]."
    assert steps == ["Searching the codebase", "Writing the answer"]


def test_ask_without_an_llm_has_no_stream(make_api_client, codebase):
    api = make_api_client()
    repo_id = indexed(api, codebase)

    answer = api.ask(repo_id, "How are passwords hashed?", provider="none")

    assert answer.stream is None
    assert answer.sources[0].metadata["source"] == "app/auth.py"


def test_http_errors_raise_with_the_apis_message(make_api_client, codebase):
    api = make_api_client()

    with pytest.raises(ApiError, match="No repository with id nope"):
        api.repo("nope")
    repo = api.add_repo(path=str(codebase))
    with pytest.raises(ApiError, match="Index this repository first"):
        list(api.chat_events(repo["id"], "anything"))


def test_error_events_raise(make_api_client, codebase):
    class FailingLLM(FakeLLM):
        def stream(self, system, messages, max_tokens=None, effort=None):
            yield "partial "
            raise RuntimeError("model crashed")

    api = make_api_client(llm=FailingLLM())
    repo_id = indexed(api, codebase)

    answer = api.ask(repo_id, "How are passwords hashed?", provider="ollama", mode="oneshot")
    with pytest.raises(ApiError, match="model crashed"):
        list(answer.stream)


def test_waiting_gives_up_after_the_timeout(make_api_client, codebase, monkeypatch):
    api = make_api_client()
    monkeypatch.setattr(api, "job", lambda job_id: {"id": job_id, "status": "running"})

    with pytest.raises(TimeoutError):
        api.wait_for_job("abc", timeout=0.05)


def test_unknown_chat_options_are_rejected(make_api_client):
    api = make_api_client()

    with pytest.raises(TypeError, match="temperature"):
        list(api.chat_events("any", "question", temperature=0.2))
