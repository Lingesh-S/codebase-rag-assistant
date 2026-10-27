from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit" / "app.py")


@pytest.fixture
def no_llm(monkeypatch, tmp_path):
    for name in ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    monkeypatch.setenv("RAG_PERSIST_DIR", str(tmp_path / "chroma"))


def test_app_renders_and_asks_for_an_index_first(no_llm):
    at = AppTest.from_file(APP, default_timeout=60).run()

    assert not at.exception
    assert at.sidebar.selectbox[0].value == "none"

    at.chat_input[0].set_value("What does this project do?").run()

    assert not at.exception
    assert "build the knowledge base" in at.chat_message[-1].markdown[0].value


def test_ollama_model_defaults_to_the_configured_model(no_llm, monkeypatch):
    from rag_pipeline import generator

    monkeypatch.setattr(generator, "ollama_models", lambda: ["qwen2.5-coder:7b", "qwen3.5:4b", "gpt-oss:20b"])
    monkeypatch.setitem(generator.DEFAULT_MODELS, "ollama", "qwen3.5:4b")
    st.cache_data.clear()  # provider detection is cached between runs

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


@pytest.mark.slow
def test_build_and_ask_in_retrieval_only_mode(no_llm):
    at = AppTest.from_file(APP, default_timeout=600).run()
    next(b for b in at.sidebar.button if "Build" in b.label).click().run()

    assert not at.exception
    assert any("Knowledge base ready" in s.value for s in at.sidebar.success)

    at.chat_input[0].set_value("How are user passwords hashed?").run()

    assert not at.exception
    assert any("Sources" in e.label for e in at.expander)
    assert "inventory/auth.py" in " ".join(m.value for m in at.markdown)
