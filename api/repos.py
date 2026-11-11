# api/repos.py

import os
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from api.jobs import Job, JobRunner, QueueJobRunner
from embeddings.embedder import get_embedding_model
from loaders.github import RepoRef, clone_or_update, parse_github_url
from rag_pipeline.pipeline import RAGPipeline, RAGSettings
from vectorstore.vector_store import IndexStats, collection_name_for


@dataclass
class Repo:
    # The id is the Chroma collection name the CLI and the Streamlit app use for the same codebase,
    # so all three share one index.
    id: str
    name: str
    kind: str  # "folder" or "github"
    location: str  # the folder path or the GitHub URL
    folder: Path | None = None  # known up front for folders, after the first clone for GitHub repos
    github: RepoRef | None = None
    status: str = "not_indexed"  # not_indexed, indexing, ready or failed
    stats: IndexStats | None = None
    job: Job | None = None
    pipeline: RAGPipeline | None = None  # the last index that finished; answers keep using it during a rebuild

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "location": self.location,
            "status": self.status,
            "stats": asdict(self.stats) if self.stats else None,
            "job_id": self.job.id if self.job else None,
        }


def build_index(
    spec: dict,
    settings: RAGSettings,
    embedding=None,
    repo_cache: str | Path = ".repos",
    report: Callable[[str, int, int], None] | None = None,
) -> tuple[RAGPipeline, IndexStats]:
    """Clones (for GitHub repos) and indexes the codebase a job spec describes; used by both job runners."""
    report = report or (lambda stage, done, total: None)
    settings = replace(settings, **spec.get("settings", {}))
    if spec["github"]:
        ref = RepoRef(**spec["github"])
        report(f"Cloning {spec['name']}", 0, 0)
        folder = clone_or_update(ref, repo_cache)
    else:
        folder = Path(spec["folder"])
    embedding = embedding or get_embedding_model(settings.embedding_model)
    rag = RAGPipeline(folder, settings, collection_name=spec["repo_id"], embedding=embedding)
    return rag, rag.build_knowledge_base(progress=report)


class RepoRegistry:
    """The codebases this API knows about, and the index jobs that build them."""

    def __init__(
        self,
        settings: RAGSettings,
        jobs: JobRunner | QueueJobRunner,
        embedding=None,
        repo_cache: str | Path = ".repos",
    ):
        self.settings = settings
        self.jobs = jobs
        self.embedding = embedding
        self.repo_cache = Path(repo_cache)
        self._repos: dict[str, Repo] = {}
        self._lock = threading.Lock()

    def add_folder(self, path: str) -> tuple[Repo, bool]:
        folder = Path(path).expanduser().resolve()
        allowed_root = os.getenv("RAG_ALLOWED_ROOT")
        if allowed_root and not folder.is_relative_to(Path(allowed_root).resolve()):
            raise ValueError(f"Only folders inside {allowed_root} can be indexed on this server.")
        if not folder.is_dir():
            raise ValueError(f"Folder not found: {path}")
        return self._add(Repo(collection_name_for(str(folder)), folder.name, "folder", str(folder), folder=folder))

    def add_github(self, url: str) -> tuple[Repo, bool]:
        ref = parse_github_url(url)
        return self._add(Repo(collection_name_for(f"github:{ref.name}"), ref.name, "github", url, github=ref))

    def get(self, repo_id: str) -> Repo | None:
        repo = self._repos.get(repo_id)
        if repo and self.jobs.remote:
            self._follow(repo)
        return repo

    def all(self) -> list[Repo]:
        return [self.get(repo_id) for repo_id in list(self._repos)]

    def index(self, repo: Repo) -> Job:
        """Starts an index job for the repo, or returns the one already running."""
        with self._lock:
            if repo.job and repo.job.active:
                return repo.job
            repo.status = "indexing"
            if self.jobs.remote:
                repo.job = self.jobs.submit(repo.id, self.job_spec(repo))
            else:
                repo.job = self.jobs.submit(repo.id, lambda job: self._build(repo, job))
            return repo.job

    def job_spec(self, repo: Repo) -> dict:
        """Everything the worker process needs to index the repo; it must survive a trip through JSON."""
        keys = ("embedding_model", "chunk_size", "chunk_overlap")
        return {
            "repo_id": repo.id,
            "name": repo.name,
            "kind": repo.kind,
            "folder": str(repo.folder) if repo.folder else None,
            "github": asdict(repo.github) if repo.github else None,
            "settings": {key: getattr(self.settings, key) for key in keys},
        }

    def _follow(self, repo: Repo) -> None:
        """Picks up the worker's progress, and opens the shared index once its job is done."""
        with self._lock:
            if repo.job is None or repo.status != "indexing":
                return
            repo.job = self.jobs.get(repo.job.id) or repo.job
            if repo.job.status == "failed":
                repo.status = "failed"
            elif repo.job.status == "done":
                repo.folder = Path(repo.job.folder) if repo.job.folder else repo.folder
                embedding = self.embedding or get_embedding_model(self.settings.embedding_model)
                repo.pipeline = RAGPipeline(repo.folder, self.settings, collection_name=repo.id, embedding=embedding)
                repo.stats, repo.status = IndexStats(**repo.job.stats), "ready"

    def _add(self, repo: Repo) -> tuple[Repo, bool]:
        with self._lock:
            if repo.id in self._repos:
                return self._repos[repo.id], False
            self._repos[repo.id] = repo
            return repo, True

    def _build(self, repo: Repo, job: Job) -> dict:
        # Runs on a worker thread.
        try:
            rag, stats = build_index(self.job_spec(repo), self.settings, self.embedding, self.repo_cache, job.report)
        except Exception:
            repo.status = "failed"
            raise
        repo.folder, job.folder = rag.codebase_path, str(rag.codebase_path)
        repo.pipeline, repo.stats, repo.status = rag, stats, "ready"
        return asdict(stats)
