import asyncio
import os
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.jobs import Job, QueueJobRunner
from api.worker import require_shared_store, run_index_job
from rag_pipeline.pipeline import RAGSettings
from tests.test_api import register, wait_for
from vectorstore.qdrant_store import QdrantStore
from vectorstore.vector_store import collection_name_for


class MemoryJobStore:
    """Stands in for RedisJobStore: job records saved as plain dicts, as they would be in Redis."""

    def __init__(self):
        self.records: dict[str, dict] = {}

    def save(self, job: Job) -> None:
        self.records[job.id] = job.to_dict()

    def load(self, job_id: str) -> Job | None:
        record = self.records.get(job_id)
        return Job(**record) if record else None


@pytest.fixture
def qdrant_settings(tmp_path, qdrant_memory) -> RAGSettings:
    return RAGSettings(vector_store="qdrant", qdrant_url=":memory:", persist_dir=None, rerank_model=None)


def folder_spec(folder, settings: RAGSettings) -> dict:
    folder = folder.resolve()
    return {
        "repo_id": collection_name_for(str(folder)),
        "name": folder.name,
        "kind": "folder",
        "folder": str(folder),
        "github": None,
        "settings": {"embedding_model": settings.embedding_model, "chunk_size": 1200, "chunk_overlap": 120},
    }


def test_job_records_round_trip_through_dicts():
    job = Job("repo-1", status="running", stage="Embedding", done=3, total=9)

    assert Job(**job.to_dict()) == job


def test_worker_job_builds_the_shared_index(codebase, qdrant_settings, fake_embeddings, tmp_path):
    jobs = MemoryJobStore()
    spec = folder_spec(codebase, qdrant_settings)
    jobs.save(Job(spec["repo_id"], id="job-1"))

    result = run_index_job("job-1", spec, jobs, qdrant_settings, fake_embeddings, tmp_path / "repos")

    assert result["status"] == "done"
    assert result["stats"]["total_files"] == 5
    assert result["folder"] == spec["folder"]
    assert jobs.load("job-1").finished_at is not None
    # Another process (the API) sees the same points.
    shared = QdrantStore(fake_embeddings, spec["repo_id"], ":memory:")
    assert shared.count() == result["stats"]["total_chunks"]


def test_worker_job_saves_progress_as_it_goes(codebase, qdrant_settings, fake_embeddings, tmp_path):
    seen = []

    class RecordingStore(MemoryJobStore):
        def save(self, job):
            seen.append((job.status, job.stage))
            super().save(job)

    run_index_job("job-2", folder_spec(codebase, qdrant_settings), RecordingStore(), qdrant_settings, fake_embeddings)

    assert seen[0][0] == "running"
    assert ("running", "Embedding") in seen
    assert seen[-1][0] == "done"


def test_worker_job_failures_are_recorded(tmp_path, qdrant_settings, fake_embeddings):
    jobs = MemoryJobStore()
    spec = folder_spec(tmp_path, qdrant_settings) | {"folder": str(tmp_path / "gone")}

    result = run_index_job("job-3", spec, jobs, qdrant_settings, fake_embeddings)

    assert result["status"] == "failed"
    assert "not found" in result["error"]
    assert jobs.load("job-3").status == "failed"


def test_worker_job_clones_github_repos(codebase, qdrant_settings, fake_embeddings, tmp_path, monkeypatch):
    import api.repos

    cloned = []
    monkeypatch.setattr(api.repos, "clone_or_update", lambda ref, cache: cloned.append((ref.name, cache)) or codebase)
    spec = folder_spec(codebase, qdrant_settings) | {
        "kind": "github",
        "folder": None,
        "github": {"owner": "pallets", "repo": "itsdangerous", "ref": None},
    }

    result = run_index_job("job-4", spec, MemoryJobStore(), qdrant_settings, fake_embeddings, tmp_path / "repos")

    assert result["status"] == "done"
    assert cloned == [("pallets/itsdangerous", tmp_path / "repos")]
    assert result["folder"] == str(codebase)


def test_worker_mode_refuses_chroma():
    with pytest.raises(RuntimeError, match="RAG_VECTOR_STORE=qdrant"):
        require_shared_store(RAGSettings(vector_store="chroma"))
    require_shared_store(RAGSettings(vector_store="qdrant"))


def test_api_refuses_to_queue_jobs_with_chroma(monkeypatch, tmp_path):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379")

    with pytest.raises(RuntimeError, match="RAG_VECTOR_STORE=qdrant"):
        create_app(RAGSettings(vector_store="chroma", persist_dir=str(tmp_path)), warm_up=False)


@pytest.fixture
def queued_client(qdrant_settings, fake_embeddings, tmp_path):
    """The API in worker mode, with a thread standing in for the worker process and Redis."""
    store = MemoryJobStore()
    enqueued = []

    def enqueue(job_id: str, spec: dict) -> None:
        enqueued.append(spec)
        threading.Thread(
            target=run_index_job, args=(job_id, spec, store, qdrant_settings, fake_embeddings, tmp_path / "repos")
        ).start()

    jobs = QueueJobRunner(store, enqueue)
    app = create_app(
        qdrant_settings, embedding=fake_embeddings, jobs=jobs, repo_cache=tmp_path / "repos", warm_up=False
    )
    with TestClient(app) as client:
        client.enqueued = enqueued
        yield client


def test_api_sends_index_jobs_to_the_worker(queued_client, codebase):
    repo_id = register(queued_client, codebase)

    job = wait_for(queued_client, queued_client.post(f"/repos/{repo_id}/index").json()["id"])

    assert job["status"] == "done"
    [spec] = queued_client.enqueued
    assert (spec["repo_id"], spec["kind"], spec["folder"]) == (repo_id, "folder", str(codebase.resolve()))
    repo = queued_client.get(f"/repos/{repo_id}").json()
    assert repo["status"] == "ready"
    assert repo["stats"]["total_chunks"] == job["stats"]["total_chunks"]
    sources = queued_client.post(f"/repos/{repo_id}/search", json={"query": "shopping cart total"}).json()
    assert sources[0]["source"] == "app/cart.py"


def test_api_reports_worker_failures(queued_client, tmp_path):
    folder = tmp_path / "gone"
    folder.mkdir()
    repo_id = register(queued_client, folder)
    folder.rmdir()

    job = wait_for(queued_client, queued_client.post(f"/repos/{repo_id}/index").json()["id"])

    assert job["status"] == "failed"
    assert queued_client.get(f"/repos/{repo_id}").json()["status"] == "failed"


def test_api_does_not_queue_a_second_job_while_one_is_active(qdrant_settings, fake_embeddings, codebase, tmp_path):
    enqueued = []
    jobs = QueueJobRunner(MemoryJobStore(), lambda job_id, spec: enqueued.append(job_id))
    app = create_app(qdrant_settings, embedding=fake_embeddings, jobs=jobs, repo_cache=tmp_path, warm_up=False)
    with TestClient(app) as client:
        repo_id = register(client, codebase)
        first = client.post(f"/repos/{repo_id}/index").json()
        second = client.post(f"/repos/{repo_id}/index").json()

        assert first["id"] == second["id"] == enqueued[0]
        assert len(enqueued) == 1
        assert client.get(f"/repos/{repo_id}").json()["status"] == "indexing"


def services_up() -> bool:
    import redis
    from qdrant_client import QdrantClient

    try:
        redis.Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379"), socket_timeout=1).ping()
        QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=1).get_collections()
    except Exception:
        return False
    return True


@pytest.mark.slow
def test_live_worker_indexes_through_redis_and_qdrant(codebase, fake_embeddings, tmp_path, monkeypatch):
    if not services_up():
        pytest.skip("Redis and Qdrant are not running (docker compose -f docker-compose.dev.yml up -d)")
    import redis
    from arq.worker import Worker

    from api import worker
    from api.jobs import RedisJobStore

    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    settings = RAGSettings(vector_store="qdrant", qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    spec = folder_spec(codebase, settings) | {"repo_id": f"test-live-{uuid.uuid4().hex[:8]}"}
    store = RedisJobStore(redis.Redis.from_url(redis_url))
    queue = worker.ArqQueue(redis_url)
    jobs = QueueJobRunner(store, queue)
    try:
        job = jobs.submit(spec["repo_id"], spec)
        assert store.load(job.id).status == "queued"

        ctx = {"jobs": store, "settings": settings, "embedding": fake_embeddings, "repo_cache": tmp_path}

        async def drain_queue():
            # arq binds a worker to the event loop that is running when it is built.
            burst = Worker(
                functions=[worker.index_repo],
                redis_settings=worker.redis_settings(redis_url),
                burst=True,
                handle_signals=False,
                ctx=ctx,
            )
            try:
                await burst.main()
            finally:
                await burst.pool.aclose()  # Worker.close() needs SIGUSR1, which Windows lacks

        asyncio.run(drain_queue())

        finished = store.load(job.id)
        assert finished.status == "done", finished.error
        assert QdrantStore(fake_embeddings, spec["repo_id"], settings.qdrant_url).count() > 0
    finally:
        jobs.shutdown()
        store.redis.delete(store.KEY.format(job.id), f"arq:result:{job.id}")
        QdrantStore(fake_embeddings, spec["repo_id"], settings.qdrant_url).client.delete_collection(spec["repo_id"])
