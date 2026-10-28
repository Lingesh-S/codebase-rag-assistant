import os
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before importing project modules: they read their defaults (models, paths) at import time.
load_dotenv(PROJECT_ROOT / ".env")

from embeddings.embedder import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL, get_embedding_model  # noqa: E402
from loaders.code_loader import SUPPORTED_EXTENSIONS, extract_zip  # noqa: E402
from loaders.github import clone_or_update, parse_github_url  # noqa: E402
from rag_pipeline.generator import DEFAULT_MODELS, available_providers, get_llm, ollama_models  # noqa: E402
from rag_pipeline.pipeline import RAGPipeline, RAGSettings  # noqa: E402
from utils.chunker import chunk_body  # noqa: E402
from vectorstore.vector_store import IndexStats, collection_name_for  # noqa: E402

PROVIDER_LABELS = {
    "huggingface": "Hugging Face Inference",
    "anthropic": "Anthropic Claude",
    "ollama": "Ollama (local)",
    "none": "None (retrieval only)",
}
CODE_LANGUAGES = {"js": "javascript", "ts": "typescript", "shell": "bash", "text": None}
SOURCE_KINDS = ["Folder path", "GitHub repository", "Upload files"]
REPO_CACHE = Path(os.getenv("RAG_REPO_CACHE", str(PROJECT_ROOT / ".repos")))
WARM_UP = os.getenv("RAG_WARM_UP", "1") != "0"

# --- Page Config ---
st.set_page_config(page_title="Codebase RAG Assistant", layout="wide", page_icon="🤖")

# --- Header + GitHub Button ---
st.markdown(
    """
    <style>
        .github-btn-fixed {
            position: absolute;
            top: -20px;
            right: 30px;
            z-index: 1000;
        }

        .github-btn-fixed a {
            background-color: #3cc2ff;
            color: #0b0b0b;
            font-weight: 600;
            padding: 8px 18px;
            border-radius: 8px;
            text-decoration: none;
            font-size: 14px;
            border: none;
        }

        .github-btn-fixed a:hover {
            background-color: #289ecf;
            color: white;
        }

        .header-centered {
            background-color: #1e1f24;
            border: 1.5px solid #3cc2ff;
            border-radius: 16px;
            padding: 25px 20px;
            margin-top: 25px;
            margin-bottom: 35px;
            text-align: center;
        }

        .header-centered h1 {
            color: #3cc2ff;
            font-size: 2.4em;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .header-centered p {
            color: #b0b3b8;
            font-size: 1.05em;
            margin: 0;
        }
    </style>

    <!-- GitHub floating button -->
    <div class="github-btn-fixed">
        <a href="https://github.com/Lingesh-S/codebase-rag-assistant" target="_blank">⭐ GitHub</a>
    </div>

    <!-- Centered header -->
    <div class="header-centered">
        <h1>🤖 Codebase RAG Assistant</h1>
        <p>Ask questions about your codebase using AI-powered document retrieval & understanding</p>
    </div>
""",
    unsafe_allow_html=True,
)


# --- Cached resources (shared by all sessions) ---
@st.cache_resource(show_spinner=False)
def warm_embedding_model(name: str) -> threading.Thread:
    """Loads the embedding model in the background at start-up, so the first index build does not wait."""
    thread = threading.Thread(target=get_embedding_model, args=(name,), daemon=True)
    thread.start()
    return thread


@st.cache_resource(show_spinner=False)
def warm_ollama_model(model: str) -> threading.Thread:
    """Loads a local model in the background as soon as it is picked, so the first answer does not wait."""
    thread = threading.Thread(target=lambda: get_llm("ollama", model).warm_up(), daemon=True)
    thread.start()
    return thread


@st.cache_data(ttl=60, show_spinner=False)
def detect_providers() -> list[str]:
    return available_providers()


@st.cache_data(ttl=60, show_spinner=False)
def local_ollama_models() -> list[str]:
    return ollama_models()


def resolve_folder(path_text: str) -> Path:
    path = Path(path_text.strip()).expanduser()
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    path = path.resolve()
    allowed_root = os.getenv("RAG_ALLOWED_ROOT")
    if allowed_root and not path.is_relative_to(Path(allowed_root).resolve()):
        raise ValueError(f"Only folders inside {allowed_root} can be indexed on this server.")
    if not path.is_dir():
        raise ValueError(f"Folder not found: {path_text}")
    return path


def save_uploads(files, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    for file in files:
        if file.name.lower().endswith(".zip"):
            extract_zip(file.getvalue(), destination)
        else:
            # Only keep the base name so a crafted file name cannot escape the folder.
            (destination / Path(file.name).name).write_bytes(file.getvalue())


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            symbol = f" · `{src['symbol']}`" if src["symbol"] else ""
            st.markdown(f"**[{src['n']}]** `{src['source']}` · lines {src['start_line']}-{src['end_line']}{symbol}")
            st.code(src["code"], language=CODE_LANGUAGES.get(src["language"], src["language"]))


def source_dicts(chunks) -> list[dict]:
    return [
        {
            "n": i,
            "source": c.metadata["source"],
            "start_line": c.metadata["start_line"],
            "end_line": c.metadata["end_line"],
            "symbol": c.metadata.get("symbol", ""),
            "language": c.metadata.get("language", "text"),
            "code": chunk_body(c.text, c.metadata.get("header_lines", 0)),
        }
        for i, c in enumerate(chunks, 1)
    ]


# --- Background indexing ---
@dataclass
class IndexJob:
    """An index build running on a background thread; the page polls it for progress."""

    label: str
    stage: str = "Starting"
    done: int = 0
    total: int = 0
    rag: RAGPipeline | None = None
    stats: IndexStats | None = None
    error: str | None = None
    finished: bool = False
    thread: threading.Thread | None = None

    def report(self, stage: str, done: int, total: int) -> None:
        self.stage, self.done, self.total = stage, done, total


def run_index_job(job: IndexJob, prepare, settings: RAGSettings, collection: str | None) -> None:
    # Runs off the script thread, so it touches only the job object, never st.*.
    try:
        folder = prepare(job)
        embedding = get_embedding_model(settings.embedding_model)
        job.rag = RAGPipeline(folder, settings, collection_name=collection, embedding=embedding)
        job.stats = job.rag.build_knowledge_base(progress=job.report)
    except Exception as exc:
        job.error, job.rag = str(exc), None
    finally:
        job.finished = True


def start_index_job(label: str, prepare, settings: RAGSettings, collection: str | None) -> None:
    job = IndexJob(label)
    job.thread = threading.Thread(target=run_index_job, args=(job, prepare, settings, collection), daemon=True)
    st.session_state.index_job = job
    job.thread.start()


def collect_finished_job() -> None:
    """Swaps a finished index in (or records its error) on the script thread."""
    job = st.session_state.index_job
    if job is None or not job.finished:
        return
    st.session_state.index_job = None
    if job.error:
        st.session_state.index_notice = ("error", job.error)
        return
    st.session_state.rag = job.rag
    device = getattr(job.rag.store.embedding, "device", "cpu").upper()
    st.session_state.index_info = (job.label, job.stats, device)
    if job.stats.total_chunks == 0:
        st.session_state.index_notice = ("warning", "No supported source files were found.")
    else:
        st.session_state.index_notice = ("success", "Index ready.")


@st.fragment(run_every=1.0)
def index_progress() -> None:
    job = st.session_state.index_job
    if job is None:
        return
    if job.finished:
        st.rerun()  # a full rerun swaps the new index in
    fraction = min(job.done / job.total, 1.0) if job.total else 0.0
    detail = f" {job.done:,}/{job.total:,}" if job.total else ""
    st.progress(fraction, text=f"{job.label} · {job.stage}{detail}")


# --- Session state ---
st.session_state.setdefault("messages", [])
st.session_state.setdefault("rag", None)
st.session_state.setdefault("index_info", None)
st.session_state.setdefault("index_job", None)
st.session_state.setdefault("index_notice", None)
st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])
collect_finished_job()
if WARM_UP:
    warm_embedding_model(DEFAULT_EMBEDDING_MODEL)


# --- Sidebar ---
with st.sidebar:
    st.header("Codebase")
    source_kind = st.radio("Index from", SOURCE_KINDS)
    uploaded_files, folder_text, repo_url = [], None, None
    if source_kind == "Folder path":
        folder_text = st.text_input("Folder path", value="sample_codebase")
    elif source_kind == "GitHub repository":
        repo_url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo")
    else:
        uploaded_files = st.file_uploader(
            "Source files or a .zip of your project",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS] + ["zip"],
            accept_multiple_files=True,
        )

    started = False
    if st.button("Build index", type="primary", width="stretch", disabled=st.session_state.index_job is not None):
        try:
            if source_kind == "Upload files":
                if not uploaded_files:
                    raise ValueError("Upload at least one file first.")
                folder = Path(tempfile.gettempdir()) / f"rag-upload-{st.session_state.session_id}"
                save_uploads(uploaded_files, folder)
                label, settings = "uploaded files", RAGSettings(persist_dir=None)
                collection = f"upload-{st.session_state.session_id}"

                def prepare(job, folder=folder):
                    return folder

            elif source_kind == "GitHub repository":
                repo = parse_github_url(repo_url or "")
                label, settings, collection = repo.name, RAGSettings(), collection_name_for(f"github:{repo.name}")

                def prepare(job, repo=repo):
                    job.report(f"Cloning {repo.name}", 0, 0)
                    return clone_or_update(repo, REPO_CACHE)

            else:
                folder = resolve_folder(folder_text or "")
                label, settings, collection = folder.name, RAGSettings(), None

                def prepare(job, folder=folder):
                    return folder

            st.session_state.index_notice = None
            start_index_job(label, prepare, settings, collection)
            started = True
        except Exception as exc:
            st.session_state.index_notice = ("error", str(exc))
    if started:
        st.rerun()

    if st.session_state.index_job is not None:
        index_progress()
    if st.session_state.index_notice:
        kind, text = st.session_state.index_notice
        getattr(st, kind)(text)
    if st.session_state.index_info:
        name, stats, device = st.session_state.index_info
        st.caption(
            f"**{name}**: {stats.total_files} files, {stats.total_chunks} chunks · "
            f"{stats.files_added} new, {stats.files_updated} changed, {stats.files_removed} removed, "
            f"{stats.files_unchanged} unchanged · embeddings on {device}"
        )

    st.header("Model")
    detected = detect_providers()
    provider_options = ["huggingface", "anthropic", "ollama", "none"]
    default_provider = detected[0] if detected else "none"
    provider = st.selectbox(
        "LLM provider",
        provider_options,
        index=provider_options.index(default_provider),
        format_func=lambda p: PROVIDER_LABELS[p] + (" ✓" if p in detected else ""),
        help="✓ marks providers configured on this machine (HF_TOKEN, ANTHROPIC_API_KEY or a running Ollama).",
    )

    model_name = None
    if provider == "ollama":
        installed = local_ollama_models()
        if installed:
            preferred = DEFAULT_MODELS["ollama"]
            index = installed.index(preferred) if preferred in installed else 0
            model_name = st.selectbox("Model", installed, index=index)
            if preferred and preferred not in installed:
                st.caption(f"OLLAMA_MODEL={preferred} is not installed, so {installed[0]} is used.")
            if WARM_UP:
                warm_ollama_model(model_name)
        else:
            st.warning("Ollama is not running or has no models. Try `ollama pull qwen3.5:4b`.")
    elif provider != "none":
        model_name = st.text_input("Model", value=DEFAULT_MODELS[provider])

    with st.expander("Advanced settings"):
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.2,
            disabled=provider in ("anthropic", "none"),
            help="Current Claude models choose their own sampling settings.",
        )
        max_tokens = st.slider(
            "Max answer tokens", 256, 8192, 2048, step=256, disabled=provider in ("anthropic", "none")
        )
        top_k = st.slider("Chunks to retrieve", 3, 12, 6)
        use_reranker = st.toggle(
            "Cross-encoder reranking",
            value=False,
            help=f"Re-orders the top hybrid results with {DEFAULT_RERANK_MODEL}.",
        )
        rewrite_followups = st.toggle(
            "Rewrite follow-up questions",
            value=True,
            help="Turns follow-ups such as 'and the title?' into standalone search queries.",
        )

    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption("[GitHub repo](https://github.com/Lingesh-S/codebase-rag-assistant)")


# --- Chat Section ---
st.markdown("### Ask a question about your codebase")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_sources(message.get("sources", []))
        st.markdown(message["content"])

query = st.chat_input("Ask about the code...")

if query:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m.get("in_history", True) and m.get("content")
    ]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    rag = st.session_state.rag
    in_history = False
    with st.chat_message("assistant"):
        if rag is None:
            building = st.session_state.index_job is not None
            content = (
                "The index is still being built; ask again when it finishes."
                if building
                else "Build the index first: pick a codebase in the sidebar and press Build index."
            )
            sources = []
            st.markdown(content)
        else:
            try:
                llm = None
                if provider != "none" and model_name:
                    llm = get_llm(provider, model_name, temperature=temperature, max_tokens=max_tokens)
                status = st.status("Searching the codebase", expanded=False)
                answer = rag.stream_answer(
                    query,
                    llm,
                    history,
                    rewrite_followups=rewrite_followups,
                    k=top_k,
                    rerank=use_reranker,
                    on_step=lambda label: status.update(label=label),
                )
                sources = source_dicts(answer.sources)
                files = len({s["source"] for s in sources})
                status.update(label=f"Found {len(sources)} passages in {files} files", state="complete")
                if answer.search_query != query:
                    st.caption(f"Searched for: _{answer.search_query}_")
                render_sources(sources)
                if answer.stream is None:
                    content = "No LLM is configured, so here is the most relevant code."
                    st.markdown(content)
                else:
                    content = st.write_stream(answer.stream)
                    in_history = True
            except Exception as exc:
                content, sources = f"The request failed: {exc}", []
                st.markdown(content)

    st.session_state.messages.append(
        {"role": "assistant", "content": content, "sources": sources, "in_history": in_history}
    )
