# rag_pipeline/main.py
#
# Usage (from the project root):
#   python -m rag_pipeline.main --path sample_codebase
#   python -m rag_pipeline.main --path ../my-project --provider ollama -q "Where is auth handled?"

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

# Load .env before importing project modules: they read their defaults (models, paths) at import time.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from rag_pipeline.generator import PROVIDERS, available_providers, get_llm  # noqa: E402
from rag_pipeline.pipeline import RAGPipeline, RAGSettings  # noqa: E402
from retriever.code_retriever import RETRIEVAL_MODES  # noqa: E402
from utils.chunker import chunk_body  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ask questions about a codebase from the terminal.")
    parser.add_argument("--path", default="sample_codebase", help="folder to index (default: sample_codebase)")
    parser.add_argument("--provider", choices=[*PROVIDERS, "none"], help="LLM backend (default: first one configured, else none)")
    parser.add_argument("--model", help="model name for the chosen provider")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("-k", type=int, default=6, help="number of chunks to retrieve (default: 6)")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid", help="retrieval strategy (default: hybrid)")
    parser.add_argument("--rerank", action="store_true", help="reorder results with a cross-encoder (slower on CPU)")
    parser.add_argument("--rebuild", action="store_true", help="drop the existing index before indexing")
    return parser.parse_args(argv)


def print_sources(sources, with_code=False):
    for i, chunk in enumerate(sources, 1):
        meta = chunk.metadata
        symbol = f"  ({meta['symbol']})" if meta.get("symbol") else ""
        print(f"  [{i}] {meta['source']}:{meta['start_line']}-{meta['end_line']}{symbol}")
        if with_code:
            code = chunk_body(chunk.text, meta.get("header_lines", 0))
            print("      " + "\n      ".join(code.splitlines()[:12]))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)

    settings = RAGSettings(top_k=args.k, retrieval_mode=args.mode)

    print(f"Indexing {Path(args.path).resolve()} ...")
    rag = RAGPipeline(args.path, settings)
    if args.rebuild:
        rag.store.reset()
    stats = rag.build_knowledge_base()
    print(
        f"Index ready: {stats.total_files} files, {stats.total_chunks} chunks "
        f"(new {stats.files_added}, changed {stats.files_updated}, removed {stats.files_removed}, unchanged {stats.files_unchanged})"
    )

    provider = args.provider or next(iter(available_providers()), "none")
    llm = None if provider == "none" else get_llm(provider, args.model)
    print(f"LLM: {provider} / {llm.model}" if llm else "LLM: none (retrieval-only mode, showing matching code)")

    history: list[dict] = []

    def ask(question: str) -> None:
        answer = rag.stream_answer(question, llm, history, rerank=args.rerank)
        if answer.search_query != question:
            print(f"(searching for: {answer.search_query})")
        if answer.stream is None:
            print("\nTop matching code:")
            print_sources(answer.sources, with_code=True)
            return
        print()
        text = ""
        for piece in answer.stream:
            print(piece, end="", flush=True)
            text += piece
        print("\n\nSources:")
        print_sources(answer.sources)
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": text}])

    def safe_ask(question: str) -> bool:
        try:
            ask(question)
            return True
        except Exception as exc:  # provider/network errors should not kill the session
            print(f"\nThe {provider} request failed: {exc}", file=sys.stderr)
            return False

    if args.question:
        sys.exit(0 if safe_ask(args.question) else 1)

    print("\nAsk questions about your codebase. Type 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            safe_ask(question)


if __name__ == "__main__":
    main()
