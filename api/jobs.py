# api/jobs.py

import json
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    """One index build. The worker thread updates it as it goes; requests only read it."""

    repo_id: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "queued"  # queued, running, done or failed
    stage: str = ""
    done: int = 0
    total: int = 0
    stats: dict | None = None
    error: str | None = None
    folder: str | None = None  # where the codebase was read from (GitHub repos are cloned first)
    created_at: str = field(default_factory=now)
    finished_at: str | None = None

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def report(self, stage: str, done: int, total: int) -> None:
        self.stage, self.done, self.total = stage, done, total

    def to_dict(self) -> dict:
        return asdict(self)


class JobRunner:
    """Runs index jobs on background threads, so a request never waits for indexing."""

    remote = False

    def __init__(self, max_workers: int = 2):
        self.jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="index-job")

    def submit(self, repo_id: str, work: Callable[[Job], dict]) -> Job:
        job = Job(repo_id)
        self.jobs[job.id] = job
        self._executor.submit(self._run, job, work)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _run(job: Job, work: Callable[[Job], dict]) -> None:
        job.status = "running"
        try:
            job.stats = work(job)
            job.status = "done"
        except Exception as exc:
            job.error, job.status = str(exc), "failed"
        finally:
            job.finished_at = now()


class RedisJobStore:
    """Job records kept in Redis as JSON: the worker writes them, the API reads them."""

    KEY = "rag:job:{}"
    TTL_SECONDS = 7 * 24 * 3600

    def __init__(self, redis):
        self.redis = redis

    def save(self, job: Job) -> None:
        self.redis.set(self.KEY.format(job.id), json.dumps(job.to_dict()), ex=self.TTL_SECONDS)

    def load(self, job_id: str) -> Job | None:
        record = self.redis.get(self.KEY.format(job_id))
        return Job(**json.loads(record)) if record else None


class QueueJobRunner:
    """
    Hands index jobs to the separate worker process (python -m api.worker) through a queue.

    A job carries everything the worker needs (see RepoRegistry.job_spec), and its progress
    is read back from the job store, so the API and the worker share no memory.
    """

    remote = True

    def __init__(self, store, enqueue: Callable[[str, dict], None]):
        self.store = store
        self.enqueue = enqueue

    def submit(self, repo_id: str, spec: dict) -> Job:
        job = Job(repo_id)
        self.store.save(job)
        self.enqueue(job.id, spec)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.store.load(job_id)

    def shutdown(self) -> None:
        close = getattr(self.enqueue, "close", None)
        if close:
            close()
