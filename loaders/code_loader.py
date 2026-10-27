# loaders/code_loader.py

import fnmatch
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

from langchain_core.documents import Document

# File extension -> language name. Names match langchain_text_splitters.Language
# values where a code-aware splitter exists; the rest fall back to plain text splitting.
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".ipynb": "python",
    ".js": "js",
    ".jsx": "js",
    ".mjs": "js",
    ".ts": "ts",
    ".tsx": "ts",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".swift": "swift",
    ".lua": "lua",
    ".sol": "sol",
    ".html": "html",
    ".md": "markdown",
    ".rst": "rst",
    ".tex": "latex",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "shell",
    ".css": "css",
    ".txt": "text",
}

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".nox", "venv", ".venv", "env",
    "node_modules", "site-packages", "dist", "build", ".next", "target",
    "chroma_db", ".ipynb_checkpoints",
}

# Never index anything that commonly holds credentials, plus generated noise.
IGNORED_FILE_PATTERNS = [
    ".env*", "secrets.*", "credentials.*", "service-account*.json", "*.min.js",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
]

MAX_FILE_BYTES = 1_000_000


def is_ignored_file(name: str) -> bool:
    lower = name.lower()
    return any(fnmatch.fnmatch(lower, pattern) for pattern in IGNORED_FILE_PATTERNS)


def iter_code_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place so os.walk never descends into ignored folders.
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and not is_ignored_file(filename):
                yield path


def read_text(path: Path, max_bytes: int = MAX_FILE_BYTES) -> str | None:
    """Returns the file's text, or None for binary, empty or oversized files."""
    if path.stat().st_size > max_bytes:
        return None
    raw = path.read_bytes()
    if not raw.strip() or b"\x00" in raw[:8192]:
        return None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    if path.suffix.lower() == ".ipynb":
        return notebook_to_text(text)
    return text


def notebook_to_text(raw_json: str) -> str | None:
    """Flattens a Jupyter notebook into Python-style text, with markdown cells as comments."""
    try:
        cells = json.loads(raw_json).get("cells", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    parts = []
    for cell in cells:
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if cell.get("cell_type") == "markdown":
            source = "\n".join(f"# {line}" for line in source.splitlines())
        parts.append(source)
    return "\n\n".join(parts) or None


def extract_zip(data: bytes, destination: str | Path, max_total_bytes: int = 200_000_000) -> None:
    """Extracts an uploaded archive, refusing entries that would land outside destination (zip slip)."""
    destination = Path(destination).resolve()
    total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe path in archive: {info.filename}")
            total += info.file_size
            if total > max_total_bytes:
                raise ValueError("Archive is too large to index")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def load_codebase(directory_path: str | Path, max_file_bytes: int = MAX_FILE_BYTES) -> list[Document]:
    """
    Loads every supported source file under directory_path as one Document per file.

    Metadata holds the path relative to the codebase root, the language and a
    content hash that the vector store uses for incremental re-indexing.
    """
    root = Path(directory_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase folder not found: {directory_path}")

    documents = []
    for path in iter_code_files(root):
        try:
            text = read_text(path, max_file_bytes)
        except OSError:
            continue
        if text is None:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.relative_to(root).as_posix(),
                    "language": SUPPORTED_EXTENSIONS[path.suffix.lower()],
                    "file_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                },
            )
        )
    return documents
