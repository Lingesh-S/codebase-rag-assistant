import pytest

from rag_pipeline.agent import MAX_TOOL_CALLS, Agent, CodebaseTools, renumber_citations
from rag_pipeline.generator import ToolCall, ToolTurn
from rag_pipeline.pipeline import Sources, Step, Token
from tests.conftest import FakeLLM, ScriptedLLM, write
from vectorstore.vector_store import StoredChunk


@pytest.fixture
def rag(make_rag, codebase):
    pipeline = make_rag(codebase)
    pipeline.build_knowledge_base()
    return pipeline


def call(name, n=1, **arguments):
    return ToolCall(f"call_{n}", name, arguments)


def run(rag, llm, question="How are passwords hashed?", **kwargs):
    events = list(Agent(rag, llm, **kwargs).answer_events(question))
    steps = [e.label for e in events if isinstance(e, Step)]
    sources = next(e for e in events if isinstance(e, Sources))
    answer = "".join(e.text for e in events if isinstance(e, Token))
    return steps, sources, answer


def tool_results(llm, request=-1) -> list[dict]:
    return [m for m in llm.tool_requests[request]["messages"] if m["role"] == "tool"]


# ---- The loop ----------------------------------------------------------------------------


def test_agent_searches_then_answers_with_the_passages_it_cited(rag):
    llm = ScriptedLLM([ToolTurn("", [call("search_code", query="password hashing")]), ToolTurn("PBKDF2 [1].", [])])

    steps, sources, answer = run(rag, llm)

    assert steps == ["Searching: password hashing"]
    assert answer == "PBKDF2 [1]."
    assert [c.metadata["source"] for c in sources.chunks] == ["app/auth.py"]
    result = tool_results(llm)[0]["content"]
    assert result.startswith("[1] app/auth.py lines ")
    assert "def hash_password" in result
    assert llm.tool_requests[0]["tools"] == ["search_code", "read_file", "list_dir", "grep"]


def test_citations_are_renumbered_to_the_sources_actually_cited(rag):
    llm = ScriptedLLM(
        [
            ToolTurn("", [call("read_file", path="app/cart.py")]),
            ToolTurn("", [call("read_file", 2, path="app/auth.py", start_line=4, end_line=6)]),
            ToolTurn("Hashing lives in hash_password [2], see also [2][99].", []),
        ]
    )

    labels, sources, answer = run(rag, llm)

    assert labels == ["Reading app/cart.py lines 1-11", "Reading app/auth.py lines 4-6"]
    assert answer == "Hashing lives in hash_password [1], see also [1][?]."
    assert len(sources.chunks) == 1
    passage = sources.chunks[0]
    assert (passage.metadata["source"], passage.metadata["start_line"], passage.metadata["end_line"]) == (
        "app/auth.py",
        4,
        6,
    )
    assert "def hash_password" in passage.text


def test_an_answer_without_citations_keeps_every_opened_passage(rag):
    llm = ScriptedLLM([ToolTurn("", [call("read_file", path="app/cart.py")]), ToolTurn("It holds items.", [])])

    _, sources, answer = run(rag, llm)

    assert answer == "It holds items."
    assert [c.metadata["source"] for c in sources.chunks] == ["app/cart.py"]


def test_tool_errors_go_back_to_the_model_and_the_loop_continues(rag):
    llm = ScriptedLLM(
        [
            ToolTurn(
                "",
                [
                    call("delete_repo", 1),
                    call("read_file", 2),
                    ToolCall("call_3", "search_code", "not json"),
                    call("read_file", 4, path="../outside.py"),
                ],
            ),
            ToolTurn("Nothing found.", []),
        ]
    )

    _, _, answer = run(rag, llm)

    results = tool_results(llm)
    assert answer == "Nothing found."
    assert all(r["is_error"] for r in results)
    assert "Unknown tool delete_repo" in results[0]["content"]
    assert "path" in results[1]["content"]
    assert "JSON object" in results[2]["content"]
    assert "outside" in results[3]["content"]


def test_the_tool_budget_ends_with_a_forced_answer(rag):
    llm = ScriptedLLM([ToolTurn("", [call("search_code", query="cart")])])  # never stops searching on its own

    events = list(Agent(rag, llm).answer_events("cart?"))

    steps = [e for e in events if isinstance(e, Step)]
    assert len([s for s in steps if s.label.startswith("Searching")]) == MAX_TOOL_CALLS
    last = llm.tool_requests[-1]
    assert last["tools"] == []
    assert "Answer now" in last["messages"][-1]["content"]


def test_calls_past_the_budget_in_one_turn_are_refused(rag):
    many = [call("list_dir", n, path="") for n in range(1, MAX_TOOL_CALLS + 3)]
    llm = ScriptedLLM([ToolTurn("", many), ToolTurn("Done.", [])])

    steps, _, _ = run(rag, llm)

    assert len(steps) == MAX_TOOL_CALLS
    results = tool_results(llm)
    assert len(results) == len(many)  # every call still gets a result
    assert results[-1]["is_error"] and "budget" in results[-1]["content"]


def test_history_comes_before_the_question(rag):
    llm = ScriptedLLM([ToolTurn("Fine.", [])])
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    list(Agent(rag, llm).answer_events("and the cart?", history))

    assert llm.tool_requests[0]["messages"] == [*history, {"role": "user", "content": "and the cart?"}]


# ---- Fallbacks ---------------------------------------------------------------------------


def test_models_without_tool_calling_get_one_shot_retrieval(rag):
    events = list(Agent(rag, FakeLLM()).answer_events("How are passwords hashed?"))

    assert [e.label for e in events if isinstance(e, Step)] == ["Searching the codebase", "Writing the answer"]
    assert "".join(e.text for e in events if isinstance(e, Token)) == "It hashes passwords with PBKDF2 [1]."


def test_a_model_that_rejects_tools_falls_back_to_one_shot(rag):
    llm = ScriptedLLM([RuntimeError("qwen does not support tools")])

    steps, sources, answer = run(rag, llm)

    assert steps[-2:] == ["Searching the codebase", "Writing the answer"]
    assert sources.chunks[0].metadata["source"] == "app/auth.py"
    assert answer == llm.reply


def test_an_empty_final_answer_is_written_from_the_opened_passages(rag):
    llm = ScriptedLLM([ToolTurn("", [call("read_file", path="app/auth.py")]), ToolTurn("  ", [])])

    steps, sources, answer = run(rag, llm)

    assert steps == ["Reading app/auth.py lines 1-7", "Writing the answer"]
    assert [c.metadata["source"] for c in sources.chunks] == ["app/auth.py"]
    assert answer == llm.reply
    assert '<source id="1" file="app/auth.py" lines="1-7">' in llm.calls[0]["messages"][-1]["content"]


def test_a_failure_mid_loop_answers_from_what_was_opened(rag):
    llm = ScriptedLLM([ToolTurn("", [call("search_code", query="password")]), RuntimeError("connection reset")])

    steps, sources, answer = run(rag, llm)

    assert steps == ["Searching: password", "Writing the answer"]
    assert sources.chunks
    assert answer == llm.reply


# ---- The tools ---------------------------------------------------------------------------


@pytest.fixture
def tools(rag):
    return CodebaseTools(rag)


def test_list_dir_shows_only_indexed_files(tools):
    assert tools.run(call("list_dir", path="")) == "README.md\napp/\nsettings.json\nweb/"
    assert tools.run(call("list_dir", path="app")) == "app/auth.py\napp/cart.py"
    assert tools.run(call("list_dir")) == tools.run(call("list_dir", path="/"))


def test_list_dir_rejects_unknown_folders(tools):
    with pytest.raises(ValueError, match="node_modules"):
        tools.run(call("list_dir", path="node_modules"))


def test_read_file_numbers_lines_and_caps_the_range(rag, codebase, tools):
    write(codebase, "app/long.py", "".join(f"x{i} = {i}\n" for i in range(1, 501)))
    rag.build_knowledge_base()
    tools = CodebaseTools(rag)

    result = tools.run(call("read_file", path="app/long.py", start_line=10))

    lines = result.splitlines()
    assert lines[0] == "[1] app/long.py lines 10-209 (of 500)"
    assert lines[1] == "10| x10 = 10"
    assert lines[-1] == "209| x209 = 209"
    assert tools.passages[0].metadata["end_line"] == 209


@pytest.mark.parametrize("path", ["../outside.py", ".env", "secrets.toml", "node_modules/lib/index.js", "nope.py"])
def test_read_file_only_opens_indexed_files(tools, path):
    with pytest.raises(ValueError):
        tools.run(call("read_file", path=path))


def test_read_file_accepts_windows_and_dotted_paths(tools):
    assert tools.run(call("read_file", path=".\\app\\cart.py")).startswith("[1] app/cart.py lines 1-")


def test_reopening_a_passage_keeps_its_number(tools):
    tools.run(call("read_file", path="app/cart.py"))
    tools.run(call("read_file", path="app/auth.py"))

    assert tools.run(call("read_file", path="app/cart.py")).startswith("[1] ")
    assert len(tools.passages) == 2


def test_grep_finds_lines_in_indexed_files(tools):
    result = tools.run(call("grep", pattern=r"def \w+_item"))

    assert result == "app/cart.py:7:     def add_item(self, sku, quantity):"


def test_grep_can_be_narrowed_to_a_folder_and_reports_no_matches(tools):
    assert "web/client.js:1:" in tools.run(call("grep", pattern="fetch", path="web"))
    assert tools.run(call("grep", pattern="fetch", path="app")) == "No matches."


def test_grep_rejects_bad_patterns(tools):
    with pytest.raises(ValueError, match="regular expression"):
        tools.run(call("grep", pattern="("))


def test_search_code_can_be_narrowed_to_a_folder(tools):
    result = tools.run(call("search_code", query="fetch retry", path="web"))

    assert result.startswith("[1] web/client.js lines ")
    assert all(p.metadata["source"].startswith("web/") for p in tools.passages)


def test_renumber_citations_keeps_first_use_order():
    passages = [StoredChunk(str(n), "", {"source": f"f{n}.py"}) for n in range(1, 5)]

    text, used = renumber_citations("See [3] and [1], again [3][7].", passages)

    assert text == "See [1] and [2], again [1][?]."
    assert [p.id for p in used] == ["3", "1"]
