from rag_pipeline.generator import SYSTEM_PROMPT
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
