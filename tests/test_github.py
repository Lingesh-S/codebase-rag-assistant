import subprocess

import pytest

from loaders import github
from loaders.github import RepoRef, clone_or_update, parse_github_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/Lingesh-S/codebase-rag-assistant", ("Lingesh-S", "codebase-rag-assistant", None)),
        ("https://github.com/psf/requests.git", ("psf", "requests", None)),
        ("https://github.com/psf/requests/", ("psf", "requests", None)),
        ("github.com/pallets/flask/tree/3.1.x", ("pallets", "flask", "3.1.x")),
        ("https://github.com/owner/repo/tree/feature/login", ("owner", "repo", "feature/login")),
    ],
)
def test_parses_github_urls(url, expected):
    ref = parse_github_url(url)
    assert (ref.owner, ref.repo, ref.ref) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/owner/repo",
        "https://github.com/owner",
        "https://github.com/-owner/repo",
        "https://github.com/owner/repo/tree/--upload-pack=evil",
        "https://github.com/owner/repo;rm -rf ~",
        "file:///etc/passwd",
        "",
    ],
)
def test_rejects_anything_that_is_not_a_plain_github_repo(url):
    with pytest.raises(ValueError):
        parse_github_url(url)


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def origin(tmp_path, monkeypatch):
    """A local repository standing in for GitHub, so clone and update run for real."""
    repo = tmp_path / "origin"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    (repo / "app.py").write_text("def hello():\n    return 'v1'\n", encoding="utf-8")
    git("add", ".", cwd=repo)
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "v1", cwd=repo)
    monkeypatch.setattr(RepoRef, "clone_url", property(lambda self: str(repo)))
    return repo


def test_clones_then_updates_to_the_latest_commit(origin, tmp_path):
    ref = parse_github_url("https://github.com/demo/app")

    path = clone_or_update(ref, tmp_path / "cache")
    assert (path / "app.py").read_text(encoding="utf-8").endswith("'v1'\n")

    (origin / "app.py").write_text("def hello():\n    return 'v2'\n", encoding="utf-8")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-am", "v2", cwd=origin)

    assert clone_or_update(ref, tmp_path / "cache") == path
    assert (path / "app.py").read_text(encoding="utf-8").endswith("'v2'\n")


def test_git_failures_become_readable_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(RepoRef, "clone_url", property(lambda self: str(tmp_path / "missing")))

    with pytest.raises(RuntimeError, match="Could not clone owner/repo"):
        clone_or_update(parse_github_url("https://github.com/owner/repo"), tmp_path / "cache")


def test_git_never_prompts_for_credentials(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        (tmp_path / "cache" / "o__r" / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(github.subprocess, "run", fake_run)
    clone_or_update(parse_github_url("https://github.com/o/r"), tmp_path / "cache")

    assert seen["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert seen["timeout"] > 0
