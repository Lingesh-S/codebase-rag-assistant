# api/app.py
#
# One HTTP service for every client: register a codebase, index it in the background, then search
# it or ask questions with the answer streamed as server-sent events. Run it with `python -m api`.

import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.sse import EventSourceResponse, format_sse_event

from api.jobs import JobRunner, QueueJobRunner, RedisJobStore
from api.repos import Repo, RepoRegistry
from api.schemas import ChatIn, RepoIn, SearchIn
from embeddings.embedder import get_embedding_model
from rag_pipeline.agent import SEARCH_K, ToolError, agent_events, run_tool
from rag_pipeline.generator import LLM, available_providers, get_llm
from rag_pipeline.pipeline import PROJECT_ROOT, RAGPipeline, RAGSettings, Sources, Step, Token
from utils.chunker import chunk_body
from vectorstore.vector_store import StoredChunk

REPO_CACHE = Path(os.getenv("RAG_REPO_CACHE", str(PROJECT_ROOT / ".repos")))
WARM_UP = os.getenv("RAG_WARM_UP", "1") != "0"


def default_llm(provider: str | None, model: str | None) -> LLM | None:
    """The named provider, else the first one configured on this machine, else none (retrieval only)."""
    provider = provider or next(iter(available_providers()), "none")
    return None if provider == "none" else get_llm(provider, model)


def warm_models(settings: RAGSettings, embedding=None) -> None:
    """Loads the embedding model and a local Ollama model before the first request needs them."""
    with suppress(Exception):
        (embedding or get_embedding_model(settings.embedding_model)).embed_query("warm up")
    with suppress(Exception):
        if next(iter(available_providers()), None) == "ollama":
            get_llm("ollama").warm_up()


def source_dict(n: int, chunk: StoredChunk) -> dict:
    meta = chunk.metadata
    return {
        "n": n,
        "source": meta["source"],
        "start_line": meta["start_line"],
        "end_line": meta["end_line"],
        "symbol": meta.get("symbol", ""),
        "language": meta.get("language", "text"),
        "code": chunk_body(chunk.text, meta.get("header_lines", 0)),
    }


def sse_message(event: str, data: dict) -> bytes:
    return format_sse_event(event=event, data_str=json.dumps(data))


def sse_answer(events: Iterator[Step | Sources | Token], llm: LLM | None) -> Iterator[bytes]:
    """Encodes answer events as server-sent events; the stream always ends with done or error."""
    search_query, answer = "", []
    try:
        for event in events:
            if isinstance(event, Step):
                yield sse_message("step", {"label": event.label})
            elif isinstance(event, Sources):
                search_query = event.search_query
                sources = [source_dict(n, chunk) for n, chunk in enumerate(event.chunks, 1)]
                yield sse_message("sources", {"search_query": search_query, "sources": sources})
            else:
                answer.append(event.text)
                yield sse_message("token", {"text": event.text})
    except Exception as exc:  # the 200 status is already sent, so a failure travels as an event
        yield sse_message("error", {"message": str(exc)})
        return
    done = {"answer": "".join(answer), "search_query": search_query}
    done |= {"provider": llm.provider, "model": llm.model} if llm else {"provider": "none", "model": None}
    yield sse_message("done", done)


def tool_result(rag: RAGPipeline, name: str, arguments: dict, k: int) -> dict:
    try:
        label, result = run_tool(rag, name, arguments, k)
    except ToolError as exc:  # a bad call is the caller's mistake, not a server failure
        raise HTTPException(400, str(exc)) from exc
    return {"label": label, "result": result}


def job_runner(settings: RAGSettings, max_workers: int) -> JobRunner | QueueJobRunner:
    """The separate worker process when REDIS_URL is set, else threads inside this process."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return JobRunner(max_workers)
    import redis

    from api.worker import ArqQueue, require_shared_store

    require_shared_store(settings)
    return QueueJobRunner(RedisJobStore(redis.Redis.from_url(redis_url)), ArqQueue(redis_url))


def create_app(
    settings: RAGSettings | None = None,
    embedding=None,
    llm_factory: Callable[[str | None, str | None], LLM | None] = default_llm,
    repo_cache: str | Path = REPO_CACHE,
    max_workers: int = 2,
    warm_up: bool = WARM_UP,
    jobs: JobRunner | QueueJobRunner | None = None,
) -> FastAPI:
    settings = settings or RAGSettings()
    jobs = jobs or job_runner(settings, max_workers)
    repos = RepoRegistry(settings, jobs, embedding, repo_cache)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm_up:
            app.state.warm_thread = threading.Thread(
                target=warm_models, args=(repos.settings, embedding), name="warm-up", daemon=True
            )
            app.state.warm_thread.start()
        yield
        jobs.shutdown()

    app = FastAPI(title="Codebase RAG Assistant", version="0.2.0", lifespan=lifespan)

    def find_repo(repo_id: str) -> Repo:
        repo = repos.get(repo_id)
        if repo is None:
            raise HTTPException(404, f"No repository with id {repo_id}")
        return repo

    def ready_index(repo_id: str) -> RAGPipeline:
        repo = find_repo(repo_id)
        if repo.pipeline is None:
            busy = repo.status == "indexing"
            raise HTTPException(409, "The first index is still building" if busy else "Index this repository first")
        return repo.pipeline

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "repos": len(repos.all())}

    @app.post("/repos", status_code=201)
    def add_repo(body: RepoIn, response: Response) -> dict:
        try:
            repo, created = repos.add_folder(body.path) if body.path is not None else repos.add_github(body.github_url)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not created:
            response.status_code = 200
        return repo.to_dict()

    @app.get("/repos")
    def list_repos() -> list[dict]:
        return [repo.to_dict() for repo in repos.all()]

    @app.get("/repos/{repo_id}")
    def get_repo(repo_id: str) -> dict:
        return find_repo(repo_id).to_dict()

    @app.post("/repos/{repo_id}/index", status_code=202)
    def index_repo(repo_id: str) -> dict:
        return repos.index(find_repo(repo_id)).to_dict()

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No job with id {job_id}")
        return job.to_dict()

    @app.post("/repos/{repo_id}/search")
    def search(repo_id: str, body: SearchIn) -> list[dict]:
        chunks = ready_index(repo_id).retrieve(body.query, body.k)
        return [source_dict(n, chunk) for n, chunk in enumerate(chunks, 1)]

    @app.post("/repos/{repo_id}/tools/{name}")
    def tool(repo_id: str, name: str, arguments: Annotated[dict | None, Body()] = None, k: int = SEARCH_K) -> dict:
        """Runs one of the agent's tools (search_code, read_file, list_dir, grep) for MCP and other clients."""
        return tool_result(ready_index(repo_id), name, arguments or {}, k)

    @app.post("/repos/{repo_id}/chat", responses={200: {"content": {"text/event-stream": {}}}})
    def chat(repo_id: str, body: ChatIn) -> EventSourceResponse:
        """Streams step, sources and token events, then done (or error)."""
        rag = ready_index(repo_id)
        try:
            llm = llm_factory(body.provider, body.model)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        history = [message.model_dump() for message in body.history]
        if body.mode == "agent" and llm is not None:
            events = agent_events(rag, body.question, llm, history, body.k, body.rerank)
        else:
            events = rag.answer_events(body.question, llm, history, body.rewrite_followups, body.k, body.rerank)
        return EventSourceResponse(sse_answer(events, llm))

    return app
