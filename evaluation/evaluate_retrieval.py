# evaluation/evaluate_retrieval.py
#
# Measures retrieval quality on a labelled question set, for each retrieval strategy.
#
#   python -m evaluation.evaluate_retrieval
#   python -m evaluation.evaluate_retrieval --embedding-model jinaai/jina-embeddings-v2-base-code

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddings.embedder import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL, get_embedding_model, get_reranker  # noqa: E402
from rag_pipeline.pipeline import RAGPipeline, RAGSettings  # noqa: E402

CONFIGS = [
    ("dense", "dense", False),
    ("bm25", "bm25", False),
    ("hybrid", "hybrid", False),
    ("hybrid + rerank", "hybrid", True),
]
CUTOFFS = (1, 3, 5)


def first_hit_rank(sources: list[str], expected: list[str]) -> int | None:
    return next((rank for rank, source in enumerate(sources, 1) if source in expected), None)


def evaluate(rag: RAGPipeline, cases: list[dict], mode: str, rerank: bool, k: int = 10) -> dict:
    rag.settings.retrieval_mode = mode
    hits = dict.fromkeys(CUTOFFS, 0)
    reciprocal_ranks = 0.0
    misses = []
    started = time.perf_counter()
    for case in cases:
        results = rag.retrieve(case["question"], k=k, rerank=rerank)
        rank = first_hit_rank([r.metadata["source"] for r in results], case["expected_sources"])
        if rank is None:
            misses.append(case["question"])
            continue
        reciprocal_ranks += 1 / rank
        for cutoff in CUTOFFS:
            hits[cutoff] += rank <= cutoff
    n = len(cases)
    return {
        **{f"hit@{c}": hits[c] / n for c in CUTOFFS},
        "mrr": reciprocal_ranks / n,
        "ms_per_query": 1000 * (time.perf_counter() - started) / n,
        "misses": misses,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare retrieval strategies on a labelled question set.")
    parser.add_argument("--codebase", default=str(ROOT / "sample_codebase"))
    parser.add_argument("--golden", default=str(ROOT / "evaluation" / "golden_set.json"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--chunk-size", type=int, default=RAGSettings.chunk_size)
    args = parser.parse_args(argv)

    cases = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    settings = RAGSettings(persist_dir=None, embedding_model=args.embedding_model, rerank_model=args.rerank_model, chunk_size=args.chunk_size)
    rag = RAGPipeline(
        args.codebase,
        settings,
        collection_name="evaluation",
        embedding=get_embedding_model(args.embedding_model),
        reranker=get_reranker(args.rerank_model),
    )
    stats = rag.build_knowledge_base()
    print(f"{len(cases)} questions | {stats.total_files} files, {stats.total_chunks} chunks")
    print(f"embedding: {args.embedding_model} | reranker: {args.rerank_model}\n")

    print(f"{'config':<17}" + "".join(f"{'hit@' + str(c):>8}" for c in CUTOFFS) + f"{'MRR':>8}{'ms/query':>10}")
    all_misses = {}
    for name, mode, rerank in CONFIGS:
        result = evaluate(rag, cases, mode, rerank)
        all_misses[name] = result["misses"]
        print(f"{name:<17}" + "".join(f"{result[f'hit@{c}']:>8.2f}" for c in CUTOFFS) + f"{result['mrr']:>8.3f}{result['ms_per_query']:>10.1f}")

    for name, misses in all_misses.items():
        for question in misses:
            print(f"  missed in top 10 ({name}): {question}")


if __name__ == "__main__":
    main()
