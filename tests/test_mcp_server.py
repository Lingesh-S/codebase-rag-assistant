import re
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters

from api.client import RagClient
from mcp_server.backends import LocalBackend, backend_from_env
from mcp_server.server import create_server
from rag_pipeline.pipeline import RAGSettings
from tests.conftest import write

ROOT = Path(__file__).resolve().parents[1]
PASSAGE_HEAD = re.compile(r"^\[\d+\] (\S+) lines (\d+)-(\d+)", re.MULTILINE)


@pytest.fixture
def local_backend(tmp_path, fake_embeddings):
    backend = LocalBackend(
        RAGSettings(persist_dir=str(tmp_path / "mcp-chroma"), rerank_model=None), fake_embeddings, tmp_path / "repos"
    )
    yield backend
    backend.close()


def session(server, *calls: tuple[str, dict]) -> list[tuple[str, bool]]:
    """Runs tool calls in order in one client session; returns each result's text and error flag."""

    async def run():
        results = []
        async with Client(server) as client:
            for name, arguments in calls:
                result = await client.call_tool(name, arguments)
                results.append(("\n".join(c.text for c in result.content), result.is_error))
        return results

    return anyio.run(run)


def test_the_server_offers_the_codebase_tools(local_backend):
    async def names():
        async with Client(create_server(local_backend)) as client:
            return {tool.name for tool in (await client.list_tools()).tools}

    assert anyio.run(names) == {"index_repo", "index_status", "search_code", "read_file", "list_dir", "grep"}


def test_a_session_indexes_a_folder_then_searches_and_reads_it(local_backend, codebase):
    indexed, status, search, read, listing, grep = session(
        create_server(local_backend),
        ("index_repo", {"repo": str(codebase)}),
        ("index_status", {}),
        ("search_code", {"query": "How are passwords hashed?", "k": 2}),
        ("read_file", {"path": "app/auth.py", "start_line": 4, "end_line": 6}),
        ("list_dir", {}),
        ("grep", {"pattern": r"def \w+\(self"}),
    )

    assert not any(is_error for _, is_error in (indexed, status, search, read, listing, grep))
    assert indexed[0].startswith("Indexed project")
    assert "5 files" in indexed[0]
    assert "project" in status[0] and "ready" in status[0]
    assert PASSAGE_HEAD.findall(search[0])[0][0] == "app/auth.py"
    assert "5| def hash_password" in read[0]
    assert listing[0].splitlines() == ["README.md", "app/", "settings.json", "web/"]
    assert "app/cart.py:4:     def __init__(self):" in grep[0]


def test_bad_calls_come_back_as_tool_errors(local_backend, codebase, tmp_path):
    before, missing, _, outside, secret = session(
        create_server(local_backend),
        ("search_code", {"query": "cart"}),
        ("index_repo", {"repo": str(tmp_path / "nope")}),
        ("index_repo", {"repo": str(codebase)}),
        ("read_file", {"path": "../.env"}),
        ("read_file", {"path": ".env"}),
    )

    assert before[1] and "index_repo" in before[0]
    assert missing[1] and "Folder not found" in missing[0]
    assert outside[1] and "outside the repository" in outside[0]
    assert secret[1] and "not a file in the index" in secret[0]


def test_several_repos_need_the_repo_argument(local_backend, codebase, tmp_path):
    other = tmp_path / "other"
    write(other, "lib/math_utils.py", "def add(a, b):\n    return a + b\n")

    _, _, unnamed, named, status = session(
        create_server(local_backend),
        ("index_repo", {"repo": str(codebase)}),
        ("index_repo", {"repo": str(other)}),
        ("list_dir", {}),
        ("list_dir", {"repo": str(other)}),
        ("index_status", {}),
    )

    assert unnamed[1] and "Several repositories" in unnamed[0]
    assert named == ("lib/", False)
    assert "project" in status[0] and "other" in status[0]


def test_index_repo_can_return_before_the_job_is_done(local_backend, codebase):
    started, status = session(
        create_server(local_backend),
        ("index_repo", {"repo": str(codebase), "wait": False}),
        ("index_status", {"repo": str(codebase)}),
    )

    assert started[0].startswith("Indexing project started")
    assert "project" in status[0]
    local_backend.wait_for_job(local_backend.repos()[0]["job_id"], timeout=30)
    assert "ready" in session(create_server(local_backend), ("index_status", {"repo": str(codebase)}))[0][0]


def test_through_the_api_search_gives_the_same_sources_as_the_web_app_and_cli(make_api_client, codebase):
    api = make_api_client()
    question = "How are passwords hashed?"

    _, search = session(
        create_server(api), ("index_repo", {"repo": str(codebase)}), ("search_code", {"query": question})
    )

    repo_id = api.repos()[0]["id"]
    expected = [(c.metadata["source"], str(c.metadata["start_line"])) for c in api.search(repo_id, question)]
    assert [(source, start) for source, start, _ in PASSAGE_HEAD.findall(search[0])] == expected


def test_the_backend_follows_rag_api_url(monkeypatch):
    monkeypatch.setenv("RAG_API_URL", "http://127.0.0.1:9")
    assert isinstance(backend_from_env(), RagClient)

    monkeypatch.delenv("RAG_API_URL")
    backend = backend_from_env()
    assert isinstance(backend, LocalBackend)
    backend.close()


def test_the_stdio_command_serves_the_tools():
    # API mode at a closed port: listing tools must not need the backend.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server"], cwd=str(ROOT), env={"RAG_API_URL": "http://127.0.0.1:9"}
    )

    async def names():
        async with Client(params) as client:
            return {tool.name for tool in (await client.list_tools()).tools}

    assert "search_code" in anyio.run(names)
