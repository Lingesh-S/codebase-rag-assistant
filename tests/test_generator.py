from types import SimpleNamespace

import pytest

from rag_pipeline import generator
from rag_pipeline.generator import (
    AnthropicLLM,
    build_messages,
    condense_question,
    format_context,
    get_llm,
)
from tests.conftest import FakeLLM
from vectorstore.vector_store import StoredChunk


def chunk(source, start, end, code, symbol=""):
    header = f"File: {source} (lines {start}-{end})" + (f"\nSymbol: {symbol}" if symbol else "")
    return StoredChunk(
        id=f"{source}:{start}",
        text=f"{header}\n\n{code}",
        metadata={"source": source, "start_line": start, "end_line": end, "symbol": symbol, "header_lines": header.count("\n") + 2},
    )


def test_context_blocks_are_numbered_and_carry_location():
    context, used = format_context([chunk("a.py", 1, 2, "def a():\n    pass", "a"), chunk("b.js", 5, 5, "let b = 1;")])

    assert '<source id="1" file="a.py" lines="1-2" symbol="a">\ndef a():\n    pass\n</source>' in context
    assert '<source id="2" file="b.js" lines="5-5">' in context
    assert "File: a.py" not in context  # the chunk header is replaced by the tag attributes
    assert len(used) == 2


def test_context_respects_character_budget_but_keeps_first_chunk():
    chunks = [chunk(f"f{i}.py", 1, 1, "x" * 500) for i in range(5)]
    _, used = format_context(chunks, max_chars=1200)
    assert len(used) == 2

    _, used = format_context(chunks[:1], max_chars=10)
    assert len(used) == 1


def test_messages_contain_history_context_and_question():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    messages, used = build_messages("What does a do?", [chunk("a.py", 1, 1, "def a(): ...")], history)

    assert messages[:2] == history
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("<context>\n<source id=\"1\"")
    assert "Question: What does a do?" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("cite them inline, e.g. [1].")
    assert len(used) == 1


def test_messages_say_when_nothing_was_retrieved():
    messages, used = build_messages("anything?", [])
    assert "no matching code was found" in messages[-1]["content"]
    assert used == []


def test_condense_question_returns_first_line_of_llm_reply():
    llm = FakeLLM("Where is the ShoppingCart total computed?\n")
    history = [{"role": "user", "content": "Explain ShoppingCart"}, {"role": "assistant", "content": "It holds items."}]

    assert condense_question(llm, "and the total?", history) == "Where is the ShoppingCart total computed?"
    assert "ShoppingCart" in llm.calls[0]["messages"][0]["content"]
    assert llm.calls[0]["effort"] == "low"


def test_condense_question_skips_llm_without_history_and_survives_errors():
    llm = FakeLLM()
    assert condense_question(llm, "q?", []) == "q?"
    assert llm.calls == []

    class Broken(FakeLLM):
        def stream(self, *args, **kwargs):
            raise RuntimeError("offline")
            yield

    assert condense_question(Broken(), "q?", [{"role": "user", "content": "x"}]) == "q?"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        get_llm("openai-compatible")


def test_available_providers_reads_environment(monkeypatch):
    monkeypatch.setattr(generator, "ollama_models", lambda: [])
    for name in ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]:
        monkeypatch.delenv(name, raising=False)
    assert generator.available_providers() == []

    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(generator, "ollama_models", lambda: ["llama3"])
    assert generator.available_providers() == ["huggingface", "anthropic", "ollama"]


class FakeAnthropicStream:
    def __init__(self, stop_reason):
        self.text_stream = iter(["Hello", " world"])
        self.stop_reason = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self.stop_reason)


def fake_anthropic(stop_reason="end_turn"):
    calls = []

    def stream(**kwargs):
        calls.append(kwargs)
        return FakeAnthropicStream(stop_reason)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    return client, calls


def test_anthropic_request_uses_current_api_shape():
    llm = AnthropicLLM(api_key="test-key")
    llm.client, calls = fake_anthropic()

    text = "".join(llm.stream("system prompt", [{"role": "user", "content": "q"}], max_tokens=512))

    assert text == "Hello world"
    request = calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["output_config"] == {"effort": "medium"}
    assert request["max_tokens"] >= 16000  # thinking shares the budget, so never lowball it
    assert "temperature" not in request


def test_anthropic_refusal_is_reported_to_the_user():
    llm = AnthropicLLM(api_key="test-key")
    llm.client, _ = fake_anthropic(stop_reason="refusal")

    assert "declined" in "".join(llm.stream("s", [{"role": "user", "content": "q"}]))
