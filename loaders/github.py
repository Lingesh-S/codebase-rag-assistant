# loaders/github.py

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9][A-Za-z0-9-]{0,38})/(?P<repo>[A-Za-z0-9._-]{1,100}?)(?:\.git)?"
    r"(?:/tree/(?P<ref>[A-Za-z0-9._/-]{1,200}))?/?$"
)
GIT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str
    ref: str | None = None  # branch or tag; None means the default branch

    @property
    def name(self) -> str:
        return f"{self.owner}/{self.repo}" + (f"@{self.ref}" if self.ref else "")

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}.git"


def parse_github_url(url: str) -> RepoRef:
    """Accepts only https://github.com/<owner>/<repo>[/tree/<branch>]; refuses anything git could read as an option."""
    match = GITHUB_URL.match(url.strip())
    if not match:
        raise ValueError("Enter a GitHub repository URL such as https://github.com/owner/repo")
    ref = match["ref"]
    if ref and (ref.startswith("-") or ".." in ref):
        raise ValueError(f"Unsupported branch name: {ref}")
    return RepoRef(match["owner"], match["repo"], ref)


def _git(args: list[str], cwd: Path | None = None) -> None:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is not installed, so GitHub repositories cannot be cloned")
    # No shell, a time limit, and no credential prompt: a private or missing repo fails fast instead of hanging.
    subprocess.run(
        [git, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def clone_or_update(ref: RepoRef, cache_dir: str | Path) -> Path:
    """Shallow-clones the repository into cache_dir, or fast-forwards an earlier clone to the latest commit."""
    target = Path(cache_dir) / "__".join(part.replace("/", "_") for part in (ref.owner, ref.repo, ref.ref) if part)
    try:
        if (target / ".git").exists():
            _git(["fetch", "--depth", "1", "origin", ref.ref or "HEAD"], cwd=target)
            _git(["reset", "--hard", "FETCH_HEAD"], cwd=target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            branch = ["--branch", ref.ref] if ref.ref else []
            _git(["clone", "--depth", "1", "--single-branch", *branch, "--", ref.clone_url, str(target)])
    except subprocess.CalledProcessError as exc:
        reason = (exc.stderr or "").strip().splitlines()[-1:] or ["git failed"]
        raise RuntimeError(f"Could not clone {ref.name}: {reason[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Cloning {ref.name} took longer than {GIT_TIMEOUT_SECONDS} s") from exc
    return target
