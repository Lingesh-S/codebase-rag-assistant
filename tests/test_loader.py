import io
import json
import zipfile

import pytest

from loaders.code_loader import extract_zip, load_codebase


def test_loads_supported_files_with_relative_paths(codebase):
    docs = load_codebase(codebase)
    by_source = {d.metadata["source"]: d for d in docs}

    assert set(by_source) == {"app/auth.py", "app/cart.py", "web/client.js", "README.md", "settings.json"}
    assert by_source["app/auth.py"].metadata["language"] == "python"
    assert by_source["web/client.js"].metadata["language"] == "js"
    assert len(by_source["README.md"].metadata["file_hash"]) == 64


def test_skips_ignored_folders_secrets_binaries_and_empty_files(codebase):
    sources = {d.metadata["source"] for d in load_codebase(codebase)}
    for skipped in [
        "node_modules/lib/index.js",
        "venv/lib/site.py",
        ".git/config.json",
        ".env",
        "secrets.toml",
        "image.py",
        "empty.py",
    ]:
        assert skipped not in sources
    assert not any("super-secret" in d.page_content for d in load_codebase(codebase))


def test_notebook_cells_are_flattened(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Analysis\n", "Loads the data"]},
            {"cell_type": "code", "source": ["import pandas as pd\n", "df = pd.read_csv('x.csv')"]},
        ]
    }
    (tmp_path / "analysis.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    [doc] = load_codebase(tmp_path)

    assert doc.metadata["language"] == "python"
    assert "# # Analysis" in doc.page_content
    assert "df = pd.read_csv('x.csv')" in doc.page_content


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_codebase(tmp_path / "nope")


def make_zip(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_extract_zip_keeps_folder_structure(tmp_path):
    extract_zip(make_zip({"proj/src/main.py": "print('hi')\n", "proj/README.md": "# Proj\n"}), tmp_path)

    assert (tmp_path / "proj/src/main.py").read_text() == "print('hi')\n"
    assert {d.metadata["source"] for d in load_codebase(tmp_path)} == {"proj/src/main.py", "proj/README.md"}


def test_extract_zip_rejects_path_traversal(tmp_path):
    target = tmp_path / "uploads"
    target.mkdir()
    with pytest.raises(ValueError, match="Unsafe path"):
        extract_zip(make_zip({"../escaped.py": "x = 1\n"}), target)
    assert not (tmp_path / "escaped.py").exists()


def git(*args, cwd):
    import subprocess

    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_repos_respect_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("htmlcov/\n*.log.json\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('tracked')\n", encoding="utf-8")
    (tmp_path / "new_module.py").write_text("print('untracked but not ignored')\n", encoding="utf-8")
    (tmp_path / "htmlcov").mkdir()
    (tmp_path / "htmlcov" / "index.html").write_text("<html>coverage report</html>\n", encoding="utf-8")
    (tmp_path / "run.log.json").write_text('{"log": true}\n', encoding="utf-8")
    git("init", "-q", cwd=tmp_path)
    git("add", "app.py", ".gitignore", cwd=tmp_path)

    sources = {d.metadata["source"] for d in load_codebase(tmp_path)}

    assert sources == {"app.py", "new_module.py"}


def test_files_over_the_size_limit_are_skipped(tmp_path):
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "dump.json").write_text('{"rows": "' + "x" * 300_000 + '"}', encoding="utf-8")

    assert {d.metadata["source"] for d in load_codebase(tmp_path)} == {"small.py"}
