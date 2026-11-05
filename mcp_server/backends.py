# mcp_server/backends.py
#
# Where the MCP tools get their answers: a running API when RAG_API_URL is set (so the MCP server shares
# its indexes with the web app and the CLI), else a registry and job runner inside this process.
# Both expose the same methods as api.client.RagClient.

import os
import time
from collections.abc import Callable
from pathlib import Path

from api.app import REPO_CACHE
from api.client import RagClient
from api.jobs import JobRunner
from api.repos import RepoRegistry
from rag_pipeline.agent import SEARCH_K, run_tool
from rag_pipeline.pipeline import RAGSettings


class LocalBackend:
    def __init__(self, settings: RAGSettings | None = None, embedding=None, repo_cache: str | Path = REPO_CACHE):
        self.jobs = JobRunner(max_workers=1)
        self.registry = RepoRegistry(settings or RAGSettings(), self.jobs, embedding, repo_cache)
        self.poll_seconds = 0.2

    def add_repo(self, path: str | None = None, github_url: str | None = None) -> dict:
        repo, _ = self.registry.add_folder(path) if github_url is None else self.registry.add_github(github_url)
        return repo.to_dict()

    def repos(self) -> list[dict]:
        return [repo.to_dict() for repo in self.registry.all()]

    def repo(self, repo_id: str) -> dict:
        return self._repo(repo_id).to_dict()

    def index(self, repo_id: str) -> dict:
        return self.registry.index(self._repo(repo_id)).to_dict()

    def job(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"No job with id {job_id}")
        return job.to_dict()

    def wait_for_job(
        self, job_id: str, on_progress: Callable[[dict], None] | None = None, timeout: float | None = None
    ) -> dict:
        deadline = None if timeout is None else time.monotonic() + timeout
        while (job := self.job(job_id))["status"] in ("queued", "running"):
            if on_progress:
                on_progress(job)
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"Index job {job_id} is still {job['status']}")
            time.sleep(self.poll_seconds)
        return job

    def tool(self, repo_id: str, name: str, arguments: dict, k: int | None = None) -> str:
        repo = self._repo(repo_id)
        if repo.pipeline is None:
            raise ValueError("The first index is still building" if repo.status == "indexing" else "Index it first")
        return run_tool(repo.pipeline, name, arguments, k or SEARCH_K)[1]

    def close(self) -> None:
        self.jobs.shutdown()

    def _repo(self, repo_id: str):
        repo = self.registry.get(repo_id)
        if repo is None:
            raise ValueError(f"No repository with id {repo_id}")
        return repo


def backend_from_env() -> RagClient | LocalBackend:
    api_url = os.getenv("RAG_API_URL")
    return RagClient(api_url) if api_url else LocalBackend()
