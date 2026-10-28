from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.conftest import FakeEmbeddings

APP = str(Path(__file__).resolve().parents[1] / "streamlit" / "app.py")


@pytest.fixture
def no_llm(monkeypatch, tmp_path):
    for name in ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    monkeypatch.setenv("RAG_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RAG_WARM_UP", "0")  # no background model loading during tests
    # The app caches provider and model detection; clear it so one test's fakes never leak into the next.
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def fake_embeddings_in_app(monkeypatch):
    from embeddings import embedder

    monkeypatch.setattr(embedder, "get_embedding_model", lambda name=None: FakeEmbeddings())


def build_index(at: AppTest) -> AppTest:
    """Clicks Build index, waits for the background job, then reruns so the page picks up the result."""
    next(b for b in at.sidebar.button if "Build" in b.label).click().run()
    job = at.session_state["index_job"]
    if job is not None:
        job.thread.join(timeout=600)
    return at.run()


def test_app_renders_and_asks_for_an_index_first(no_llm):
    at = AppTest.from_file(APP, default_timeout=60).run()

    assert not at.exception
    assert at.sidebar.selectbox[0].value == "none"

    at.chat_input[0].set_value("What does this project do?").run()

    assert not at.exception
    assert "Build the index first" in at.chat_message[-1].markdown[0].value


def test_ollama_model_defaults_to_the_configured_model(no_llm, monkeypatch):
    from rag_pipeline import generator

    monkeypatch.setattr(generator, "ollama_models", lambda: ["qwen2.5-coder:7b", "qwen3.5:4b", "gpt-oss:20b"])
    monkeypatch.setitem(generator.DEFAULT_MODELS, "ollama", "qwen3.5:4b")

    at = AppTest.from_file(APP, default_timeout=60).run()

    assert not at.exception
    assert at.sidebar.selectbox[0].value == "ollama"
    assert at.sidebar.selectbox[1].value == "qwen3.5:4b"


def test_invalid_folder_shows_an_error(no_llm):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].set_value("does/not/exist").run()
    next(b for b in at.sidebar.button if "Build" in b.label).click().run()

    assert not at.exception
    assert any("Folder not found" in e.value for e in at.sidebar.error)
    assert at.session_state["index_job"] is None


def test_invalid_github_url_shows_an_error(no_llm):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("GitHub repository").run()
    at.sidebar.text_input[0].set_value("https://gitlab.com/owner/repo").run()
    next(b for b in at.sidebar.button if "Build" in b.label).click().run()

    assert not at.exception
    assert any("GitHub repository URL" in e.value for e in at.sidebar.error)


def test_index_builds_in_the_background_and_answers_with_sources(no_llm, fake_embeddings_in_app):
    at = AppTest.from_file(APP, default_timeout=60).run()

    at = build_index(at)

    assert not at.exception
    assert any("Index ready" in s.value for s in at.sidebar.success)
    assert "11 files" in " ".join(c.value for c in at.sidebar.caption)

    at.chat_input[0].set_value("How are user passwords hashed?").run()

    assert not at.exception
    assert any(e.label.startswith("Sources") for e in at.expander)
    assert "inventory/auth.py" in " ".join(m.value for m in at.markdown)


@pytest.mark.slow
def test_build_and_ask_with_the_real_models(no_llm):
    at = build_index(AppTest.from_file(APP, default_timeout=600).run())

    assert not at.exception
    assert any("Index ready" in s.value for s in at.sidebar.success)

    at.chat_input[0].set_value("How are user passwords hashed?").run()

    assert not at.exception
    assert any(e.label.startswith("Sources") for e in at.expander)
    assert "inventory/auth.py" in " ".join(m.value for m in at.markdown)
