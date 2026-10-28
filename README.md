# 🧠 Codebase RAG Assistant — AI-Powered Code Explainer
A Retrieval-Augmented Generation (RAG) pipeline that ingests, indexes and answers questions about software codebases.
Explore your own project files with natural language queries — understand, debug, and document your code effortlessly.


---

## 🎯 Objective

- Loads source files in 30+ languages, notebooks, Markdown and config files
- Splits code along its structure (functions, classes, methods) with exact line numbers
- Indexes chunks incrementally, so re-indexing only touches files that changed
- Retrieves with hybrid search (BM25 + vectors) and optional cross-encoder reranking
- Answers with an LLM that cites the exact `file:line` ranges it used
- Provides a Streamlit chat UI and a terminal CLI

---


## 🔧 Tech Stack
- **Streamlit** — chat UI with streamed answers and expandable sources
- **ChromaDB** — persistent vector store (cosine HNSW)
- **fastembed** — ONNX embedding and reranking models on CPU, no PyTorch needed
- **rank-bm25** — keyword search with a code-aware tokenizer
- **LangChain** — document model and language-aware text splitters
- **LLM providers** — Hugging Face Inference Providers, Anthropic Claude, or a local Ollama model



---

## 🏗️ How it works

```text
            ┌──────────────┐   ┌────────────────────────┐   ┌───────────────────────┐
 codebase ─▶│ code_loader  │──▶│ chunker                │──▶│ vector_store.sync()   │
            │ filters,     │   │ Python: AST functions, │   │ content-hash diff:    │
            │ hashes files │   │ classes, methods       │   │ add / update / delete │
            └──────────────┘   │ others: language-aware │   └──────────┬────────────┘
                               │ + File/Symbol header   │              │
                               └────────────────────────┘              ▼
 question ─▶ follow-up rewrite ─▶ dense top-20 ─┐                 ChromaDB
                                  BM25 top-20 ──┴▶ RRF fusion ─▶ (optional) rerank ─▶ top-k
                                                                                          │
                       answer with [n] citations ◀── LLM ◀── grounded prompt with <source> blocks
```

| Stage | What it does |
|---|---|
| Loading | Walks the folder, skips `.git`, virtualenvs, `node_modules`, build output, lock files, binaries and secret files (`.env*`, `secrets.*`, `credentials.*`). |
| Chunking | Python is parsed with `ast` so each function, class or method stays whole; large classes are split per method (`Class.method`). Other languages use LangChain's language-aware separators. Each chunk keeps its line range and gets a `File / Symbol` header that is embedded with it. |
| Indexing | Chunk IDs are deterministic and each file's SHA-256 is stored, so rebuilding only re-embeds changed files and removes deleted ones. Changing the embedding model rebuilds the collection automatically. |
| Retrieval | Dense search and BM25 (with camelCase / snake_case splitting and light stemming) are fused with Reciprocal Rank Fusion. An optional cross-encoder reorders the top 10. |
| Generation | Retrieved code is wrapped in numbered `<source>` tags, the model is told to treat it as data, cite `[n]`, and say when the context is insufficient. Follow-up questions are rewritten into standalone queries before retrieval. |

---


## 📂 Project Structure

```bash
codebase-rag-assistant/
├── streamlit/app.py         # Streamlit chat UI
├── rag_pipeline/
│   ├── pipeline.py          # RAGPipeline: index, retrieve, answer
│   ├── generator.py         # prompts + Hugging Face / Anthropic / Ollama clients
│   └── main.py              # terminal CLI
├── loaders/code_loader.py   # file discovery, filtering, notebook + zip support
├── utils/chunker.py         # AST / language-aware chunking with line numbers
├── embeddings/embedder.py   # fastembed embedding model + cross-encoder reranker
├── vectorstore/vector_store.py  # Chroma collection with incremental sync
├── retriever/code_retriever.py  # BM25 + dense hybrid retrieval, RRF, reranking
├── evaluation/              # labelled questions + retrieval benchmark
├── tests/                   # pytest suite
├── sample_codebase/         # demo project used by the app, tests and evaluation
├── requirements.txt
└── requirements-dev.txt
```

---

## 💻 Installation

Python 3.11 or newer (developed and tested on 3.12).

```bash
git clone https://github.com/Lingesh-S/codebase-rag-assistant.git
cd codebase-rag-assistant
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

**Optional: NVIDIA GPU.** Embedding every chunk is the slowest part of indexing on a CPU. To run it on an NVIDIA GPU (a driver with CUDA 13 support), swap ONNX Runtime for its GPU build after installing the requirements:

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]==1.31.1"
```

The embedding model and reranker then use the GPU automatically, and fall back to the CPU if CUDA cannot start. The sidebar and the CLI show which one is in use; set `RAG_EMBEDDING_DEVICE=cpu` to force the CPU.

Then configure at least one LLM provider by copying `.env.example` to `.env`:

| Provider | Setting | Default model |
|---|---|---|
| Hugging Face Inference Providers | `HF_TOKEN` (or the older `HUGGINGFACEHUB_API_TOKEN`) | `openai/gpt-oss-120b` |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| Ollama (local, free) | a running Ollama with a pulled model | first installed model |

Without any provider the app still works in retrieval-only mode and shows the most relevant code.
The first index build downloads the embedding model (~70 MB) once; the optional reranker (~80 MB) downloads the first time it is enabled.

---

## ▶️ How to Use

**Web app**

```bash
streamlit run streamlit/app.py
```

1. Pick a folder path (defaults to `sample_codebase`) or upload files / a `.zip` of your project.
2. Click **Build / Update Knowledge Base**. Click it again after editing code: only changed files are re-embedded.
3. Ask questions in the chat. Answers stream in with `[n]` citations; expand **Sources** to see the exact code.

**Terminal**

```bash
python -m rag_pipeline.main --path sample_codebase
python -m rag_pipeline.main --path ../my-project --provider ollama -q "Where is authentication handled?"
```

Options: `--provider {huggingface,anthropic,ollama,none}`, `--model`, `-k`, `--mode {hybrid,dense,bm25}`, `--rerank`, `--rebuild`.

---

## ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed text model, e.g. `jinaai/jina-embeddings-v2-base-code` for large codebases |
| `RAG_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Any fastembed cross-encoder |
| `RAG_EMBEDDING_DEVICE` | `auto` | `auto` uses an NVIDIA GPU when `onnxruntime-gpu` is installed; `cpu` forces the CPU |
| `RAG_PERSIST_DIR` | `chroma_db` | Where the vector index is stored |
| `RAG_ALLOWED_ROOT` | unset | Restricts the folder-path input to one directory tree. Set this on public deployments. |
| `HF_MODEL`, `ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT`, `OLLAMA_MODEL`, `OLLAMA_HOST` | see `.env.example` | Provider overrides |

---

## 📊 Evaluation

`evaluation/golden_set.json` holds 20 labelled questions about `sample_codebase` (natural-language questions and exact identifier lookups). The benchmark indexes the sample project and scores each retrieval strategy:

```bash
python -m evaluation.evaluate_retrieval
```

Results on the sample project (hit@k = an expected file is in the top k; MRR = mean reciprocal rank):

| Pipeline | hit@1 | hit@3 | MRR |
|---|---|---|---|
| Previous design: 1000-char splits, `all-MiniLM-L6-v2`, dense only | 0.75 | 0.90 | 0.839 |
| Previous splits with `bge-small-en-v1.5` | 0.85 | 1.00 | 0.917 |
| Structure-aware chunks + headers, dense only | 0.90 | 1.00 | 0.950 |
| **Structure-aware chunks + hybrid BM25/dense (default)** | **0.95** | **1.00** | **0.975** |
| Default + cross-encoder reranking | 0.95 | 1.00 | 0.975 |

Reranking adds no gain on this small, already well-ranked set and costs a few seconds per question on CPU, so it is opt-in (sidebar toggle or `--rerank`). With the weaker `all-MiniLM-L6-v2` embeddings it lifts hybrid MRR from 0.938 to 0.975, so it is worth enabling for large codebases or lighter embedding models. Extend `golden_set.json` with questions about your own code to tune these choices.

---

## 🧪 Tests and code style

```bash
pip install -r requirements-dev.txt
pytest              # fast offline suite (fake embeddings, fake LLMs)
pytest -m slow      # end-to-end app test with the real models
ruff check .        # lint (rules in ruff.toml)
ruff format .       # format to the project style (120-character lines)
```

---

## ☁️ Deploying on Streamlit Community Cloud

- Main file: `streamlit/app.py`. Choose Python 3.12 under **Advanced settings**.
- Add `HF_TOKEN` (or `ANTHROPIC_API_KEY`) under **Secrets**.
- Set `RAG_ALLOWED_ROOT` to the app folder so visitors cannot index other paths on the server. Uploaded files are indexed in memory per session.

---

## 📜 License

MIT License © 2025 Lingesh S
