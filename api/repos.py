# api/repos.py

import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from api.jobs import Job, JobRunner
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


class RepoRegistry:
    """The codebases this API knows about, and the index jobs that build them."""

    def __init__(self, settings: RAGSettings, jobs: JobRunner, embedding=None, repo_cache: str | Path = ".repos"):
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
        return self._repos.get(repo_id)

    def all(self) -> list[Repo]:
        return list(self._repos.values())

    def index(self, repo: Repo) -> Job:
        """Starts an index job for the repo, or returns the one already running."""
        with self._lock:
            if repo.job and repo.job.active:
                return repo.job
            repo.status = "indexing"
            repo.job = self.jobs.submit(repo.id, lambda job: self._build(repo, job))
            return repo.job

    def _add(self, repo: Repo) -> tuple[Repo, bool]:
        with self._lock:
            if repo.id in self._repos:
                return self._repos[repo.id], False
            self._repos[repo.id] = repo
            return repo, True

    def _build(self, repo: Repo, job: Job) -> dict:
        # Runs on a worker thread.
        try:
            if repo.github:
                job.report(f"Cloning {repo.name}", 0, 0)
                repo.folder = clone_or_update(repo.github, self.repo_cache)
            embedding = self.embedding or get_embedding_model(self.settings.embedding_model)
            rag = RAGPipeline(repo.folder, self.settings, collection_name=repo.id, embedding=embedding)
            stats = rag.build_knowledge_base(progress=job.report)
        except Exception:
            repo.status = "failed"
            raise
        repo.pipeline, repo.stats, repo.status = rag, stats, "ready"
        return asdict(stats)
