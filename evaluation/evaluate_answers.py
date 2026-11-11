# evaluation/evaluate_answers.py
#
# Compares the agent with one-shot retrieval on cross-file questions, scoring the answers' citations:
# how many of the files a good answer needs it cites, and whether every [n] points at a real, relevant source.
#
#   python evaluation/evaluate_answers.py                                # Ollama with OLLAMA_MODEL or qwen3.5:4b
#   python evaluation/evaluate_answers.py --provider anthropic --modes agent
#   python evaluation/evaluate_answers.py --limit 3 --out results.json

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from embeddings.embedder import get_embedding_model  # noqa: E402
from rag_pipeline.agent import agent_events  # noqa: E402
from rag_pipeline.generator import PROVIDERS, get_llm  # noqa: E402
from rag_pipeline.pipeline import RAGPipeline, RAGSettings, Sources, Step, Token  # noqa: E402

MODES = ("agent", "oneshot")
CITATION = re.compile(r"\[(\d+|\?)\]")


def score(answer: str, sources: list[str], expected: list[str], related: list[str] = ()) -> dict:
    """
    sources are the files of the numbered sources, in order. A citation is valid when its number
    exists and its file is one a good answer would use (expected or related).
    """
    citations = CITATION.findall(answer)
    relevant = set(expected) | set(related)
    cited_files = {sources[int(c) - 1] for c in citations if c.isdigit() and 1 <= int(c) <= len(sources)}
    valid = sum(c.isdigit() and 1 <= int(c) <= len(sources) and sources[int(c) - 1] in relevant for c in citations)
    return {
        "citations": len(citations),
        "valid_citations": valid,
        "file_recall": len(set(expected) & cited_files) / len(expected),
        "cited_files": sorted(cited_files),
    }


def ask(rag: RAGPipeline, question: str, llm, mode: str) -> dict:
    started = time.perf_counter()
    events = agent_events(rag, question, llm) if mode == "agent" else rag.answer_events(question, llm)
    steps, sources, tokens = [], [], []
    for event in events:
        if isinstance(event, Step):
            steps.append(event.label)
        elif isinstance(event, Sources):
            sources = [chunk.metadata["source"] for chunk in event.chunks]
        elif isinstance(event, Token):
            tokens.append(event.text)
    return {"answer": "".join(tokens), "sources": sources, "steps": steps, "seconds": time.perf_counter() - started}


def summarize(rows: list[dict]) -> dict:
    citations = sum(r["citations"] for r in rows)
    return {
        "questions": len(rows),
        "file_recall": sum(r["file_recall"] for r in rows) / len(rows),
        "citation_validity": sum(r["valid_citations"] for r in rows) / citations if citations else 0.0,
        "answers_with_citations": sum(r["citations"] > 0 for r in rows),
        "seconds_per_question": sum(r["seconds"] for r in rows) / len(rows),
    }


def evaluate(cases: list[dict], llm, modes=MODES, settings: RAGSettings | None = None, embedding=None) -> dict:
    settings = settings or RAGSettings(persist_dir=None, rerank_model=None)
    pipelines: dict[str, RAGPipeline] = {}
    rows: dict[str, list[dict]] = {mode: [] for mode in modes}
    for case in cases:
        repo = case.get("repo", ".")
        if repo not in pipelines:
            pipelines[repo] = RAGPipeline(
                ROOT / repo, settings, collection_name=f"answers-{len(pipelines)}", embedding=embedding
            )
            pipelines[repo].build_knowledge_base()
        for mode in modes:
            result = ask(pipelines[repo], case["question"], llm, mode)
            result |= score(result["answer"], result["sources"], case["expected_files"], case.get("related_files", []))
            rows[mode].append({"question": case["question"], **result})
            print(
                f"[{mode:<7}] recall {result['file_recall']:.2f}  valid {result['valid_citations']}/"
                f"{result['citations']}  {result['seconds']:5.1f}s  {case['question'][:70]}",
                flush=True,
            )
    return {mode: {"summary": summarize(mode_rows), "rows": mode_rows} for mode, mode_rows in rows.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score agent and one-shot answers on cross-file questions.")
    parser.add_argument("--questions", default=str(ROOT / "evaluation" / "agent_questions.json"))
    parser.add_argument("--provider", choices=PROVIDERS, default="ollama")
    parser.add_argument("--model", default=None, help="default: the provider's default (OLLAMA_MODEL for Ollama)")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--out", default=None, help="write every answer and score to this JSON file")
    args = parser.parse_args(argv)

    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))[: args.limit]
    llm = get_llm(args.provider, args.model)
    settings = RAGSettings(persist_dir=None, rerank_model=None)
    print(f"{len(cases)} questions | {llm.provider} {llm.model}\n")
    results = evaluate(cases, llm, args.modes, settings, get_embedding_model(settings.embedding_model))

    print(f"\n{'mode':<9}{'file recall':>12}{'valid citations':>17}{'cited answers':>15}{'s/question':>12}")
    for mode, result in results.items():
        s = result["summary"]
        print(
            f"{mode:<9}{s['file_recall']:>12.2f}{s['citation_validity']:>17.2f}"
            f"{s['answers_with_citations']:>9}/{s['questions']:<5}{s['seconds_per_question']:>12.1f}"
        )
    if args.out:
        payload = {"provider": llm.provider, "model": llm.model, "results": results}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
