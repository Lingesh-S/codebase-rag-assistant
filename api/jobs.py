# api/jobs.py

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
