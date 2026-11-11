import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from loaders.github import parse_github_url
from rag_pipeline.generator import ToolCall, ToolTurn
from rag_pipeline.pipeline import RAGSettings
from tests.conftest import FakeEmbeddings, FakeLLM, ScriptedLLM
from vectorstore.vector_store import collection_name_for


class BlockingEmbeddings(FakeEmbeddings):
    """Holds the first embedding call until the test releases it, so a job can be caught mid-run."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def embed_documents(self, texts):
        self.started.set()
        assert self.release.wait(10)
        return super().embed_documents(texts)


class FailingLLM(FakeLLM):
    def stream(self, system, messages, max_tokens=None, effort=None):
        yield "partial "
        raise RuntimeError("model crashed")


@pytest.fixture
def make_client(tmp_path, fake_embeddings):
    def factory(embedding=None, llm=None) -> TestClient:
        app = create_app(
            RAGSettings(persist_dir=str(tmp_path / "chroma"), rerank_model=None),
            embedding=embedding or fake_embeddings,
            llm_factory=lambda provider, model: None if provider == "none" else (llm or FakeLLM()),
            repo_cache=tmp_path / "repos",
            warm_up=False,
        )
        return TestClient(app)

    return factory


@pytest.fixture
def client(make_client):
    with make_client() as test_client:
        yield test_client


def register(client: TestClient, folder) -> str:
    response = client.post("/repos", json={"path": str(folder)})
    assert response.status_code in (200, 201), response.text
    return response.json()["id"]


def wait_for(client: TestClient, job_id: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while (job := client.get(f"/jobs/{job_id}").json())["status"] in ("queued", "running"):
        assert time.monotonic() < deadline, f"job still {job['status']}"
        time.sleep(0.05)
    return job


def indexed(client: TestClient, folder) -> str:
    repo_id = register(client, folder)
    job = wait_for(client, client.post(f"/repos/{repo_id}/index").json()["id"])
    assert job["status"] == "done", job
    return repo_id


def sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_health_reports_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_registering_a_folder_gives_a_stable_id(client, codebase):
    first = client.post("/repos", json={"path": str(codebase)})
    second = client.post("/repos", json={"path": str(codebase)})

    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["id"] == second.json()["id"] == collection_name_for(str(codebase.resolve()))
    assert first.json()["status"] == "not_indexed"
    assert [r["id"] for r in client.get("/repos").json()] == [first.json()["id"]]


def test_github_repos_share_the_web_apps_index(client):
    url = "https://github.com/pallets/itsdangerous"

    response = client.post("/repos", json={"github_url": url})

    assert response.status_code == 201
    assert response.json()["id"] == collection_name_for(f"github:{parse_github_url(url).name}")
    assert response.json()["name"] == "pallets/itsdangerous"


def test_registering_a_missing_folder_is_rejected(client, tmp_path):
    response = client.post("/repos", json={"path": str(tmp_path / "nope")})

    assert response.status_code == 400
    assert "Folder not found" in response.json()["detail"]


def test_registering_a_non_github_url_is_rejected(client):
    response = client.post("/repos", json={"github_url": "https://gitlab.com/owner/repo"})

    assert response.status_code == 400
    assert "GitHub repository URL" in response.json()["detail"]


@pytest.mark.parametrize("body", [{}, {"path": ".", "github_url": "https://github.com/a/b"}])
def test_registering_needs_exactly_one_source(client, body):
    assert client.post("/repos", json=body).status_code == 422


def test_index_job_builds_the_index_in_the_background(client, codebase):
    repo_id = register(client, codebase)

    response = client.post(f"/repos/{repo_id}/index")
    job = wait_for(client, response.json()["id"])

    assert response.status_code == 202
    assert job["status"] == "done"
    assert job["stats"]["total_files"] == 5
    repo = client.get(f"/repos/{repo_id}").json()
    assert repo["status"] == "ready"
    assert repo["stats"]["total_chunks"] == job["stats"]["total_chunks"]


def test_failed_index_jobs_report_the_error(client, tmp_path):
    folder = tmp_path / "gone"
    folder.mkdir()
    repo_id = register(client, folder)
    folder.rmdir()

    job = wait_for(client, client.post(f"/repos/{repo_id}/index").json()["id"])

    assert job["status"] == "failed"
    assert "not found" in job["error"]
    assert client.get(f"/repos/{repo_id}").json()["status"] == "failed"


def test_indexing_never_blocks_the_api(make_client, codebase):
    embedding = BlockingEmbeddings()
    with make_client(embedding=embedding) as client:
        repo_id = register(client, codebase)
        job = client.post(f"/repos/{repo_id}/index").json()
        assert embedding.started.wait(10)

        # The job is stuck inside embedding, yet every endpoint still answers at once.
        assert client.get("/health").status_code == 200
        assert client.get(f"/jobs/{job['id']}").json()["status"] == "running"
        assert client.post(f"/repos/{repo_id}/index").json()["id"] == job["id"]
        assert client.post(f"/repos/{repo_id}/chat", json={"question": "x"}).status_code == 409

        embedding.release.set()
        assert wait_for(client, job["id"])["status"] == "done"


def test_chat_streams_steps_sources_and_tokens(client, codebase):
    repo_id = indexed(client, codebase)

    response = client.post(f"/repos/{repo_id}/chat", json={"question": "How are user passwords hashed?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[:3] == ["step", "sources", "step"]
    assert set(names[3:-1]) == {"token"}
    assert names[-1] == "done"
    top = events[1][1]["sources"][0]
    assert top["source"] == "app/auth.py"
    assert top["start_line"] >= 1
    assert "hash_password" in top["code"]
    tokens = "".join(data["text"] for name, data in events if name == "token")
    assert tokens == events[-1][1]["answer"] == "It hashes passwords with PBKDF2 [1]."


def test_chat_rewrites_follow_ups_from_history(client, codebase):
    repo_id = indexed(client, codebase)
    history = [{"role": "user", "content": "Show the cart"}, {"role": "assistant", "content": "See app/cart.py"}]

    response = client.post(f"/repos/{repo_id}/chat", json={"question": "and its total?", "history": history})

    events = sse_events(response.text)
    assert events[0] == ("step", {"label": "Rewriting the follow-up question"})
    assert events[-1][1]["search_query"] != "and its total?"


def test_chat_without_an_llm_sends_sources_then_done(client, codebase):
    repo_id = indexed(client, codebase)

    response = client.post(
        f"/repos/{repo_id}/chat", json={"question": "Where is the shopping cart?", "provider": "none"}
    )

    events = sse_events(response.text)
    assert [name for name, _ in events] == ["step", "sources", "done"]
    assert events[1][1]["sources"][0]["source"] == "app/cart.py"
    assert events[-1][1]["answer"] == ""


def test_chat_reports_llm_failures_as_an_error_event(make_client, codebase):
    with make_client(llm=FailingLLM()) as client:
        repo_id = indexed(client, codebase)

        response = client.post(f"/repos/{repo_id}/chat", json={"question": "How are passwords hashed?"})

    events = sse_events(response.text)
    assert ("token", {"text": "partial "}) in events
    assert events[-1] == ("error", {"message": "model crashed"})


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x", "k": 0}, {"question": "x", "provider": "nope"}])
def test_chat_requests_are_validated(client, codebase, body):
    repo_id = indexed(client, codebase)

    assert client.post(f"/repos/{repo_id}/chat", json=body).status_code == 422


def test_search_returns_ranked_sources(client, codebase):
    repo_id = indexed(client, codebase)

    response = client.post(f"/repos/{repo_id}/search", json={"query": "shopping cart total quantity", "k": 3})

    assert response.status_code == 200
    sources = response.json()
    assert 1 <= len(sources) <= 3
    assert sources[0]["source"] == "app/cart.py"


def test_tools_endpoint_runs_the_agents_tools(client, codebase):
    repo_id = indexed(client, codebase)

    read = client.post(
        f"/repos/{repo_id}/tools/read_file", json={"path": "app/auth.py", "start_line": 4, "end_line": 6}
    )
    listing = client.post(f"/repos/{repo_id}/tools/list_dir", json={"path": "app"})
    search = client.post(f"/repos/{repo_id}/tools/search_code?k=1", json={"query": "shopping cart"})

    assert read.status_code == 200
    assert read.json()["label"] == "Reading app/auth.py lines 4-6"
    assert "5| def hash_password" in read.json()["result"]
    assert listing.json()["result"].splitlines() == ["app/auth.py", "app/cart.py"]
    assert search.json()["result"].startswith("[1] app/cart.py lines")
    assert "[2]" not in search.json()["result"]


def test_tools_endpoint_rejects_bad_calls(client, codebase):
    repo_id = indexed(client, codebase)

    outside = client.post(f"/repos/{repo_id}/tools/read_file", json={"path": "../.env"})
    secret = client.post(f"/repos/{repo_id}/tools/read_file", json={"path": ".env"})
    unknown = client.post(f"/repos/{repo_id}/tools/delete_file", json={"path": "app/auth.py"})

    assert outside.status_code == secret.status_code == unknown.status_code == 400
    assert "outside the repository" in outside.json()["detail"]
    assert "not a file in the index" in secret.json()["detail"]
    assert "Unknown tool" in unknown.json()["detail"]


def test_tools_need_an_index(client, codebase):
    repo_id = register(client, codebase)

    assert client.post(f"/repos/{repo_id}/tools/list_dir", json={}).status_code == 409


def test_unknown_repos_and_jobs_are_404(client):
    assert client.get("/repos/nope").status_code == 404
    assert client.post("/repos/nope/index").status_code == 404
    assert client.post("/repos/nope/chat", json={"question": "x"}).status_code == 404
    assert client.post("/repos/nope/search", json={"query": "x"}).status_code == 404
    assert client.get("/jobs/nope").status_code == 404


def test_python_m_api_serves_the_app_factory(monkeypatch):
    import uvicorn

    from api import __main__ as entry

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))

    entry.main(["--port", "9001"])

    assert calls == [("api.app:create_app", {"factory": True, "host": "127.0.0.1", "port": 9001})]


def test_folders_outside_the_allowed_root_are_rejected(client, codebase, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_ALLOWED_ROOT", str(tmp_path / "elsewhere"))

    response = client.post("/repos", json={"path": str(codebase)})

    assert response.status_code == 400
    assert "Only folders inside" in response.json()["detail"]


def test_done_names_the_model_that_answered(client, codebase):
    repo_id = indexed(client, codebase)

    answered = sse_events(client.post(f"/repos/{repo_id}/chat", json={"question": "cart"}).text)[-1]
    retrieval_only = sse_events(
        client.post(f"/repos/{repo_id}/chat", json={"question": "cart", "provider": "none"}).text
    )[-1]

    assert (answered[1]["provider"], answered[1]["model"]) == ("fake", "fake-model")
    assert (retrieval_only[1]["provider"], retrieval_only[1]["model"]) == ("none", None)


def test_models_warm_up_in_the_background_when_the_server_starts(monkeypatch, tmp_path):
    import api.app as app_module

    warmed = []
    monkeypatch.setattr(app_module, "warm_models", lambda settings, embedding: warmed.append(settings))
    app = create_app(RAGSettings(persist_dir=str(tmp_path / "chroma")), warm_up=True)

    with TestClient(app):
        app.state.warm_thread.join(5)

    assert len(warmed) == 1


def tool_llm():
    return ScriptedLLM(
        [
            ToolTurn("", [ToolCall("c1", "read_file", {"path": "app/auth.py", "start_line": "5"})]),
            ToolTurn("hash_password uses PBKDF2 [1].", []),
        ]
    )


def test_chat_runs_the_agent_by_default(make_client, codebase):
    llm = tool_llm()
    with make_client(llm=llm) as client:
        repo_id = indexed(client, codebase)

        response = client.post(f"/repos/{repo_id}/chat", json={"question": "How are passwords hashed?"})

    events = sse_events(response.text)
    assert [name for name, _ in events[:2]] == ["step", "sources"]
    assert events[0][1] == {"label": "Reading app/auth.py lines 5-7"}
    source = events[1][1]["sources"][0]
    assert (source["n"], source["source"], source["start_line"], source["end_line"]) == (1, "app/auth.py", 5, 7)
    assert source["code"].startswith("def hash_password")
    assert events[-1][0] == "done"
    assert events[-1][1]["answer"] == "hash_password uses PBKDF2 [1]."


def test_chat_oneshot_mode_skips_the_tools(make_client, codebase):
    llm = tool_llm()
    with make_client(llm=llm) as client:
        repo_id = indexed(client, codebase)

        response = client.post(
            f"/repos/{repo_id}/chat", json={"question": "How are passwords hashed?", "mode": "oneshot"}
        )

    assert sse_events(response.text)[0] == ("step", {"label": "Searching the codebase"})
    assert llm.tool_requests == []


def test_chat_rejects_unknown_modes(client, codebase):
    repo_id = indexed(client, codebase)

    assert client.post(f"/repos/{repo_id}/chat", json={"question": "x", "mode": "magic"}).status_code == 422
