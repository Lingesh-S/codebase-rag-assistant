# api/worker.py
#
# The index worker: builds indexes in its own process, so a big repo never slows the API down.
# The API queues jobs in Redis when REDIS_URL is set; this process picks them up and writes into
# the shared Qdrant index.
#
# Usage (from the project root, with Redis and Qdrant running, see docker-compose.dev.yml):
#   python -m api.worker
#   arq api.worker.WorkerSettings

import asyncio
import os
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from arq import create_pool  # noqa: E402
from arq.connections import RedisSettings  # noqa: E402

from api.jobs import Job, RedisJobStore, now  # noqa: E402
from api.repos import build_index  # noqa: E402
from embeddings.embedder import get_embedding_model  # noqa: E402
from rag_pipeline.pipeline import PROJECT_ROOT, RAGSettings  # noqa: E402

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
REPO_CACHE = Path(os.getenv("RAG_REPO_CACHE", str(PROJECT_ROOT / ".repos")))


def require_shared_store(settings: RAGSettings) -> None:
    if settings.vector_store != "qdrant":
        raise RuntimeError(
            "Worker mode (REDIS_URL) needs RAG_VECTOR_STORE=qdrant: Chroma cannot take writes from two processes. "
            "Start Qdrant with `docker compose -f docker-compose.dev.yml up -d`, or unset REDIS_URL to index "
            "inside the API."
        )


def redis_settings(url: str = REDIS_URL) -> RedisSettings:
    return RedisSettings.from_dsn(url)


def run_index_job(
    job_id: str, spec: dict, jobs, settings: RAGSettings, embedding=None, repo_cache: str | Path = REPO_CACHE
) -> dict:
    """Indexes the codebase a job spec describes, saving the job record as it goes; never raises."""
    job = jobs.load(job_id) or Job(spec["repo_id"], id=job_id)
    job.status = "running"
    jobs.save(job)

    def report(stage: str, done: int, total: int) -> None:
        job.report(stage, done, total)
        jobs.save(job)

    try:
        rag, stats = build_index(spec, settings, embedding, repo_cache, report)
        job.stats, job.folder, job.status = asdict(stats), str(rag.codebase_path), "done"
    except Exception as exc:
        job.error, job.status = str(exc), "failed"
    job.finished_at = now()
    jobs.save(job)
    return job.to_dict()


async def index_repo(ctx: dict, job_id: str, spec: dict) -> dict:
    # Embedding is blocking GPU/CPU work, so it runs on a thread and the worker keeps its Redis heartbeat.
    return await asyncio.to_thread(
        run_index_job, job_id, spec, ctx["jobs"], ctx["settings"], ctx.get("embedding"), ctx["repo_cache"]
    )


async def startup(ctx: dict) -> None:
    import redis

    settings = RAGSettings()
    require_shared_store(settings)
    ctx["settings"] = settings
    ctx["repo_cache"] = REPO_CACHE
    ctx["jobs"] = RedisJobStore(redis.Redis.from_url(REDIS_URL))
    ctx["embedding"] = get_embedding_model(settings.embedding_model)
    ctx["embedding"].embed_query("warm up")


class WorkerSettings:
    functions = (index_repo,)
    on_startup = startup
    redis_settings = redis_settings()
    max_jobs = 1  # one embedding model on one GPU; parallel jobs would only contend for it
    job_timeout = 3 * 3600
    handle_signals = sys.platform != "win32"  # asyncio has no signal handlers on Windows; Ctrl+C still stops it


class ArqQueue:
    """Enqueues index jobs from the API's synchronous request handlers through a private event loop."""

    def __init__(self, url: str = REDIS_URL):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="job-queue", daemon=True).start()
        self._pool = self._call(create_pool(redis_settings(url)))

    def _call(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=10)

    def __call__(self, job_id: str, spec: dict) -> None:
        self._call(self._pool.enqueue_job("index_repo", job_id, spec, _job_id=job_id))

    def close(self) -> None:
        self._call(self._pool.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)


def main() -> None:
    from arq import run_worker

    require_shared_store(RAGSettings())
    run_worker(WorkerSettings)


if __name__ == "__main__":
    main()
