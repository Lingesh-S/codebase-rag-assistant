from rag_pipeline.generator import SYSTEM_PROMPT
from rag_pipeline.pipeline import Sources, Step, Token
from tests.conftest import FakeLLM


def test_retrieval_only_mode_returns_sources_without_stream(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()

    answer = rag.stream_answer("How are passwords hashed?")

    assert answer.stream is None
    assert answer.sources[0].metadata["source"] == "app/auth.py"


def test_answer_sends_grounded_prompt_and_streams_reply(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()
    llm = FakeLLM("It uses PBKDF2 [1].")

    text, answer = rag.answer("How are passwords hashed?", llm)

    assert text == "It uses PBKDF2 [1]."
    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    prompt = call["messages"][-1]["content"]
    assert 'file="app/auth.py"' in prompt and "hash_password" in prompt
    assert prompt.count("<source ") == len(answer.sources)


def test_follow_up_questions_are_rewritten_before_retrieval(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()

    class RewritingLLM(FakeLLM):
        def stream(self, system, messages, max_tokens=None, effort=None):
            self.calls.append({"system": system, "messages": messages})
            yield "How does fetchWithRetry retry requests?" if len(self.calls) == 1 else "answer"

    llm = RewritingLLM()
    history = [{"role": "user", "content": "Show the JS client"}, {"role": "assistant", "content": "See web/client.js"}]
    answer = rag.stream_answer("how does it retry?", llm, history)
    "".join(answer.stream)

    assert answer.search_query == "How does fetchWithRetry retry requests?"
    assert answer.sources[0].metadata["source"] == "web/client.js"
    assert llm.calls[1]["messages"][:2] == history


def test_rewrite_can_be_disabled(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()
    llm = FakeLLM()

    answer = rag.stream_answer("and the cart?", llm, [{"role": "user", "content": "x"}], rewrite_followups=False)
    "".join(answer.stream)

    assert answer.search_query == "and the cart?"
    assert len(llm.calls) == 1


def test_importing_the_package_does_not_read_settings_before_env_is_loaded():
    # `python -m rag_pipeline.main` imports the package first; if that pulled in the generator,
    # model defaults would be read before main.py loads .env and OLLAMA_MODEL would be ignored.
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, rag_pipeline; print('rag_pipeline.generator' in sys.modules); "
        "from rag_pipeline import RAGPipeline; print(RAGPipeline.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,  # the assertion below reports stderr on failure
    )

    assert result.stdout.split() == ["False", "RAGPipeline"], result.stderr


def test_build_reports_its_stages_in_order(codebase, make_rag):
    rag = make_rag(codebase)
    events = []

    stats = rag.build_knowledge_base(progress=lambda stage, done, total: events.append((stage, done, total)))

    stages = list(dict.fromkeys(stage for stage, _, _ in events))
    assert stages == ["Reading files", "Splitting code", "Embedding", "Building keyword index"]
    embedding = [(done, total) for stage, done, total in events if stage == "Embedding"]
    assert embedding[-1] == (stats.chunks_added, stats.chunks_added)


def test_answers_report_their_steps(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()
    steps = []

    answer = rag.stream_answer("and the cart?", FakeLLM(), [{"role": "user", "content": "x"}], on_step=steps.append)
    "".join(answer.stream)
    rag.stream_answer("Where is the shopping cart?", on_step=steps.append)

    assert steps == [
        "Rewriting the follow-up question",
        "Searching the codebase",
        "Writing the answer",
        "Searching the codebase",
    ]


def test_answer_events_arrive_in_order_steps_sources_then_tokens(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()

    events = list(rag.answer_events("How are passwords hashed?", FakeLLM("It uses PBKDF2 [1].")))

    assert events[0] == Step("Searching the codebase")
    assert isinstance(events[1], Sources)
    assert events[1].search_query == "How are passwords hashed?"
    assert events[1].chunks[0].metadata["source"] == "app/auth.py"
    assert events[2] == Step("Writing the answer")
    assert all(isinstance(e, Token) for e in events[3:])
    assert "".join(e.text for e in events[3:]) == "It uses PBKDF2 [1]."


def test_answer_events_end_at_sources_without_an_llm(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()

    events = list(rag.answer_events("Where is the shopping cart?"))

    assert [type(e) for e in events] == [Step, Sources]
    assert events[1].chunks[0].metadata["source"] == "app/cart.py"


def test_answer_events_report_the_rewrite_before_searching(codebase, make_rag):
    rag = make_rag(codebase)
    rag.build_knowledge_base()
    history = [{"role": "user", "content": "Show the cart"}, {"role": "assistant", "content": "See app/cart.py"}]

    events = rag.answer_events("and its total?", FakeLLM("What is the cart total?"), history)

    assert next(events) == Step("Rewriting the follow-up question")
    assert next(events) == Step("Searching the codebase")
    assert next(events).search_query == "What is the cart total?"
