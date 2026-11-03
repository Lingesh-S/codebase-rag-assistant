import json

from evaluation import evaluate_answers
from evaluation.evaluate_answers import score
from rag_pipeline.generator import ToolCall, ToolTurn
from rag_pipeline.pipeline import RAGSettings
from tests.conftest import FakeEmbeddings, ScriptedLLM


def test_score_counts_expected_files_and_valid_citations():
    sources = ["a.py", "b.py", "notes.md"]

    result = score("Uses a [1] and b [2], see [3] and [9][?].", sources, ["a.py", "b.py", "c.py"], ["d.py"])

    assert result["citations"] == 5
    assert result["valid_citations"] == 2  # [3] is irrelevant, [9] does not exist, [?] was already invalid
    assert result["file_recall"] == 2 / 3
    assert result["cited_files"] == ["a.py", "b.py", "notes.md"]


def test_score_of_an_answer_without_citations_is_zero():
    assert score("No idea.", ["a.py"], ["a.py"])["file_recall"] == 0


def test_question_set_names_files_that_exist():
    cases = json.loads((evaluate_answers.ROOT / "evaluation" / "agent_questions.json").read_text(encoding="utf-8"))

    assert len(cases) == 10
    for case in cases:
        folder = evaluate_answers.ROOT / case["repo"]
        for path in case["expected_files"] + case["related_files"]:
            assert (folder / path).is_file(), path


def test_both_modes_run_and_are_scored():
    cases = [
        {
            "repo": "sample_codebase",
            "question": "How is the total inventory value computed?",
            "expected_files": ["inventory/service.py", "inventory/models.py"],
        }
    ]
    turns = [
        ToolTurn("", [ToolCall("c1", "read_file", {"path": "inventory/service.py", "start_line": 54, "end_line": 57})]),
        ToolTurn("", [ToolCall("c2", "read_file", {"path": "inventory/models.py", "start_line": 38, "end_line": 40})]),
        ToolTurn("It sums value_cents [1] from each product [2].", []),
    ]
    llm = ScriptedLLM(turns, reply="It sums prices [1].")

    results = evaluate_answers.evaluate(
        cases, llm, settings=RAGSettings(persist_dir=None, rerank_model=None), embedding=FakeEmbeddings()
    )

    agent = results["agent"]
    assert agent["summary"]["file_recall"] == 1.0
    assert agent["summary"]["citation_validity"] == 1.0
    assert agent["rows"][0]["steps"] == [
        "Reading inventory/service.py lines 54-57",
        "Reading inventory/models.py lines 38-40",
    ]
    assert results["oneshot"]["summary"]["questions"] == 1
    assert results["oneshot"]["rows"][0]["answer"] == "It sums prices [1]."
