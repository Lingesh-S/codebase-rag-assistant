from types import SimpleNamespace

import pytest
from ollama import ResponseError

from rag_pipeline import generator
from rag_pipeline.generator import (
    AnthropicLLM,
    HuggingFaceLLM,
    OllamaLLM,
    ToolCall,
    anthropic_messages,
    build_messages,
    condense_question,
    format_context,
    get_llm,
    ollama_messages,
    openai_messages,
    parse_arguments,
)
from tests.conftest import FakeLLM
from vectorstore.vector_store import StoredChunk


def chunk(source, start, end, code, symbol=""):
    header = f"File: {source} (lines {start}-{end})" + (f"\nSymbol: {symbol}" if symbol else "")
    return StoredChunk(
        id=f"{source}:{start}",
        text=f"{header}\n\n{code}",
        metadata={
            "source": source,
            "start_line": start,
            "end_line": end,
            "symbol": symbol,
            "header_lines": header.count("\n") + 2,
        },
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
    assert messages[-1]["content"].startswith('<context>\n<source id="1"')
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
    monkeypatch.setattr(generator, "ollama_models", list)
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


# The exact error Ollama returned when gpt-oss:20b crashed while loading on a 6 GB laptop GPU.
LOAD_CRASH = (
    "llama-server process has terminated: exit status 0xc0000409: The system detected an overrun of a "
    "stack-based buffer in this application.: CUDA error: shared object initialization failed"
)


class FakeOllamaClient:
    """Mimics ollama.Client.chat(stream=True): errors surface while iterating, like the real client."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.requests: list[dict] = []

    def chat(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)

        def parts():
            if isinstance(outcome, Exception):
                raise outcome
            for item in outcome:
                if isinstance(item, Exception):
                    raise item
                yield SimpleNamespace(message=SimpleNamespace(content=item))

        return parts()


def make_ollama(outcomes):
    llm = OllamaLLM(model="gpt-oss:20b", host="http://127.0.0.1:9")
    llm.client = FakeOllamaClient(outcomes)
    return llm


def ask(llm):
    return "".join(llm.stream("system", [{"role": "user", "content": "explain writer.py module"}]))


def test_ollama_retries_once_when_the_model_crashes_while_loading():
    llm = make_ollama([ResponseError(LOAD_CRASH, 500), ["The writer", " agent"]])

    assert ask(llm) == "The writer agent"
    assert llm.client.calls == 2


def test_ollama_explains_what_to_do_when_the_model_keeps_crashing():
    llm = make_ollama([ResponseError(LOAD_CRASH, 500), ResponseError(LOAD_CRASH, 500)])

    with pytest.raises(RuntimeError) as error:
        ask(llm)

    message = str(error.value)
    assert "gpt-oss:20b" in message
    assert "memory" in message and "smaller model" in message
    assert "stack-based buffer" not in message
    assert llm.client.calls == 2


def test_ollama_does_not_retry_once_the_answer_has_started():
    llm = make_ollama([["Partial", ResponseError("connection reset", 500)]])
    received = []

    with pytest.raises(ResponseError):
        for piece in llm.stream("system", [{"role": "user", "content": "q"}]):
            received.append(piece)

    assert received == ["Partial"]
    assert llm.client.calls == 1


def test_ollama_does_not_retry_client_errors():
    llm = make_ollama([ResponseError("model 'nope' not found", 404)])

    with pytest.raises(ResponseError):
        ask(llm)
    assert llm.client.calls == 1


def hf_chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def test_huggingface_streams_chat_completion_with_the_system_prompt():
    calls = []

    def chat_completion(**kwargs):
        calls.append(kwargs)
        # Providers send keep-alive chunks with no choices or empty deltas; they must be skipped.
        return iter([SimpleNamespace(choices=[]), hf_chunk("Uses "), hf_chunk(None), hf_chunk("PBKDF2 [1].")])

    llm = HuggingFaceLLM(token="hf_test", temperature=0.3, max_tokens=900)
    llm.client = SimpleNamespace(chat_completion=chat_completion)

    text = "".join(llm.stream("system prompt", [{"role": "user", "content": "q"}]))

    assert text == "Uses PBKDF2 [1]."
    request = calls[0]
    assert request["model"] == "openai/gpt-oss-120b"
    assert request["messages"][0] == {"role": "system", "content": "system prompt"}
    assert request["messages"][1] == {"role": "user", "content": "q"}
    assert (request["max_tokens"], request["temperature"], request["stream"]) == (900, 0.3, True)


def test_huggingface_token_prefers_hf_token_over_the_legacy_name(monkeypatch):
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "legacy")
    monkeypatch.setenv("HF_TOKEN", "current")
    assert generator.huggingface_token() == "current"

    monkeypatch.delenv("HF_TOKEN")
    assert generator.huggingface_token() == "legacy"


def test_get_llm_builds_each_provider_and_drops_temperature_for_claude():
    assert isinstance(get_llm("huggingface", "some/model", token="hf_test"), HuggingFaceLLM)
    assert isinstance(get_llm("ollama", "qwen3.5:4b", host="http://127.0.0.1:9"), OllamaLLM)
    claude = get_llm("anthropic", api_key="test-key", temperature=0.7)
    assert isinstance(claude, AnthropicLLM)
    assert not hasattr(claude, "temperature")


FOLLOW_UP_HISTORY = [
    {"role": "user", "content": "How does the writer agent build its prompt?"},
    {"role": "assistant", "content": "It joins the research notes into one prompt."},
]


@pytest.mark.parametrize(
    "question", ["and the title?", "How does it handle errors?", "What about the second one?", "why?", "Show me those"]
)
def test_follow_ups_that_refer_back_are_rewritten(question):
    llm = FakeLLM("How does the writer agent generate the title?")

    assert condense_question(llm, question, FOLLOW_UP_HISTORY) == "How does the writer agent generate the title?"
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "question",
    ["Where are agent metrics logged?", "How does LLMProvider choose a model?", "Explain the CLI entry point"],
)
def test_standalone_questions_skip_the_rewrite(question):
    llm = FakeLLM()

    assert condense_question(llm, question, FOLLOW_UP_HISTORY) == question
    assert llm.calls == []


def test_rewrite_asks_for_a_short_answer_with_low_effort():
    llm = FakeLLM("A standalone question?")

    condense_question(llm, "and the title?", FOLLOW_UP_HISTORY)

    assert llm.calls[0]["effort"] == "low"
    assert llm.calls[0]["max_tokens"] <= 128


def test_ollama_turns_reasoning_off_for_low_effort_calls():
    llm = make_ollama([["Rewritten?"], ["Full answer"]])

    "".join(llm.stream("system", [{"role": "user", "content": "q"}], max_tokens=128, effort="low"))
    "".join(llm.stream("system", [{"role": "user", "content": "q"}]))

    quick, normal = llm.client.requests
    assert quick["think"] is False
    assert quick["options"]["num_predict"] == 128
    assert "think" not in normal  # answers keep the model's own reasoning setting


def test_huggingface_low_effort_calls_keep_room_for_hosted_reasoning():
    calls = []

    def chat_completion(**kwargs):
        calls.append(kwargs)
        return iter([hf_chunk("ok")])

    llm = HuggingFaceLLM(token="hf_test", max_tokens=900)
    llm.client = SimpleNamespace(chat_completion=chat_completion)

    "".join(llm.stream("s", [{"role": "user", "content": "q"}], max_tokens=128, effort="low"))
    "".join(llm.stream("s", [{"role": "user", "content": "q"}], max_tokens=300))

    assert calls[0]["max_tokens"] >= 1024  # gpt-oss reasons before it answers, even for a one-line rewrite
    assert calls[1]["max_tokens"] == 300


def test_ollama_warm_up_loads_the_model_with_the_answer_settings():
    llm = make_ollama([[]])

    llm.warm_up()

    [request] = llm.client.requests
    assert request["messages"] == []
    assert request["options"]["num_ctx"] == llm.num_ctx  # a different window would force a reload later
    assert request["keep_alive"] == llm.keep_alive


def test_ollama_answers_keep_the_model_loaded():
    llm = make_ollama([["ok"]])

    "".join(llm.stream("system", [{"role": "user", "content": "q"}]))

    assert llm.client.requests[0]["keep_alive"] == llm.keep_alive


def test_ollama_defaults_to_the_ipv4_loopback(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    # "localhost" tries IPv6 first on Windows, and every call waited ~2.4 s for it while Ollama listens on 127.0.0.1.
    assert generator.ollama_host() == "http://127.0.0.1:11434"


# ---- Tool calling ----------------------------------------------------------------------

SEARCH_TOOL = {
    "name": "search_code",
    "description": "Search the code",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}

# A finished tool exchange in the provider-neutral shape the agent keeps.
TOOL_EXCHANGE = [
    {"role": "user", "content": "How are passwords hashed?"},
    {
        "role": "assistant",
        "content": "Let me look.",
        "tool_calls": [
            ToolCall("call_1", "search_code", {"query": "hash password"}),
            ToolCall("call_2", "read_file", {"path": "a.py"}),
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "search_code", "content": "[1] a.py", "is_error": False},
    {"role": "tool", "tool_call_id": "call_2", "name": "read_file", "content": "Error: no such file", "is_error": True},
]


def test_tool_exchange_converts_to_anthropic_blocks():
    messages = anthropic_messages(TOOL_EXCHANGE)

    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "call_1", "name": "search_code", "input": {"query": "hash password"}},
            {"type": "tool_use", "id": "call_2", "name": "read_file", "input": {"path": "a.py"}},
        ],
    }
    # Every result of one turn goes back in a single user message.
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "[1] a.py"},
            {"type": "tool_result", "tool_use_id": "call_2", "content": "Error: no such file", "is_error": True},
        ],
    }
    assert len(messages) == 3


def test_anthropic_assistant_turns_are_sent_back_unchanged():
    raw = [SimpleNamespace(type="thinking", thinking="", signature="sig")]
    exchange = [TOOL_EXCHANGE[0], {**TOOL_EXCHANGE[1], "raw": raw}]

    assert anthropic_messages(exchange)[1] == {"role": "assistant", "content": raw}


def test_tool_exchange_converts_to_openai_and_ollama_messages():
    openai = openai_messages(TOOL_EXCHANGE)
    assert openai[1]["tool_calls"][0] == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_code", "arguments": '{"query": "hash password"}'},
    }
    assert openai[2] == {"role": "tool", "tool_call_id": "call_1", "content": "[1] a.py"}

    ollama = ollama_messages(TOOL_EXCHANGE)
    assert ollama[1]["tool_calls"][1] == {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}
    assert ollama[3] == {"role": "tool", "tool_name": "read_file", "content": "Error: no such file"}


def test_anthropic_tool_turn_returns_calls_and_raw_content():
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="Searching."),
        SimpleNamespace(type="tool_use", id="toolu_1", name="search_code", input={"query": "auth"}),
    ]
    requests = []

    class Stream(FakeAnthropicStream):
        def get_final_message(self):
            return SimpleNamespace(stop_reason="tool_use", content=content)

    def stream(**kwargs):
        requests.append(kwargs)
        return Stream("tool_use")

    llm = AnthropicLLM(api_key="test-key")
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))

    turn = llm.call_tools("system", [{"role": "user", "content": "q"}], [SEARCH_TOOL])

    assert turn.text == "Searching."
    assert turn.calls == [ToolCall("toolu_1", "search_code", {"query": "auth"})]
    assert turn.raw is content
    assert requests[0]["tools"] == [
        {"name": "search_code", "description": "Search the code", "input_schema": SEARCH_TOOL["parameters"]}
    ]
    assert requests[0]["fallbacks"] == "default"
    assert "tool_choice" not in requests[0]  # forced tool use is rejected by current Claude models

    llm.call_tools("system", [{"role": "user", "content": "q"}], [])
    assert "tools" not in requests[1]


def test_huggingface_tool_turn_parses_json_arguments():
    requests = []

    def chat_completion(**kwargs):
        requests.append(kwargs)
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="search_code", arguments='{"query": "auth"}'))
        message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    llm = HuggingFaceLLM(token="hf_test")
    llm.client = SimpleNamespace(chat_completion=chat_completion)

    turn = llm.call_tools("system", [{"role": "user", "content": "q"}], [SEARCH_TOOL])

    assert turn.text == ""
    assert turn.calls == [ToolCall("c1", "search_code", {"query": "auth"})]
    assert requests[0]["messages"][0] == {"role": "system", "content": "system"}
    assert requests[0]["tools"][0]["function"]["name"] == "search_code"
    assert "stream" not in requests[0]


def test_tool_arguments_that_are_not_json_objects_are_kept_as_text():
    assert parse_arguments('{"query": "x"}') == {"query": "x"}
    assert parse_arguments({"query": "x"}) == {"query": "x"}
    assert parse_arguments("not json") == "not json"
    assert parse_arguments("[1, 2]") == "[1, 2]"


def test_ollama_tool_turn_numbers_its_calls():
    requests = []

    class Client:
        def chat(self, **kwargs):
            requests.append(kwargs)
            calls = [
                SimpleNamespace(function=SimpleNamespace(name="list_dir", arguments={"path": ""})),
                SimpleNamespace(function=SimpleNamespace(name="grep", arguments={"pattern": "def"})),
            ]
            return SimpleNamespace(message=SimpleNamespace(content="", tool_calls=calls))

    llm = OllamaLLM(model="qwen3.5:4b", host="http://127.0.0.1:9")
    llm.client = Client()

    turn = llm.call_tools("system", [{"role": "user", "content": "q"}], [SEARCH_TOOL])

    assert [(c.id, c.name) for c in turn.calls] == [("call_1", "list_dir"), ("call_2", "grep")]
    assert requests[0]["tools"][0] == {"type": "function", "function": SEARCH_TOOL}
    assert requests[0]["keep_alive"] == llm.keep_alive
    assert requests[0]["options"]["num_ctx"] == llm.num_ctx
