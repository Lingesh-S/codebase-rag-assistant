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
- **FastAPI** — HTTP API with background index jobs and answers streamed as server-sent events
- **ChromaDB** — persistent vector store (cosine HNSW), the default with no server needed
- **Qdrant** — optional shared vector store with hybrid dense + keyword search in one query
- **arq + Redis** — optional index worker, so large repositories index outside the API process
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
| Agent | In the API's default mode the model calls tools to search, read, list and grep the indexed files (native tool calling on all three providers), then answers citing the passages it opened. |
| Generation | Retrieved code is wrapped in numbered `<source>` tags, the model is told to treat it as data, cite `[n]`, and say when the context is insufficient. Follow-up questions are rewritten into standalone queries before retrieval. |

---


## 📂 Project Structure

```bash
codebase-rag-assistant/
├── streamlit/app.py         # Streamlit chat UI
├── api/                     # FastAPI service: repos, index jobs, search, streamed chat
│   └── worker.py            # arq index worker (used when REDIS_URL is set)
├── rag_pipeline/
│   ├── pipeline.py          # RAGPipeline: index, retrieve, answer
│   ├── agent.py             # tool-calling agent: search_code, read_file, list_dir, grep
│   ├── generator.py         # prompts + Hugging Face / Anthropic / Ollama clients
│   └── main.py              # terminal CLI
├── loaders/code_loader.py   # file discovery, filtering, notebook + zip support
├── utils/chunker.py         # AST / language-aware chunking with line numbers
├── embeddings/embedder.py   # fastembed embedding model + cross-encoder reranker
├── vectorstore/vector_store.py  # Chroma collection with incremental sync
├── vectorstore/qdrant_store.py  # Qdrant collection: dense + sparse vectors, hybrid query
├── retriever/code_retriever.py  # BM25 + dense hybrid retrieval, RRF, reranking
├── evaluation/              # labelled questions, retrieval benchmark, agent vs one-shot answer eval
├── tests/                   # pytest suite
├── sample_codebase/         # demo project used by the app, tests and evaluation
├── docker-compose.dev.yml   # Qdrant and Redis for local development
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

1. Pick a folder path (defaults to `sample_codebase`), a public GitHub repository URL, or upload files / a `.zip` of your project.
2. Click **Build index**. Click it again after editing code: only changed files are re-embedded.
3. Ask questions in the chat. Answers stream in with `[n]` citations; expand **Sources** to see the exact code.

**Terminal**

```bash
python -m rag_pipeline.main --path sample_codebase
python -m rag_pipeline.main --path ../my-project --provider ollama -q "Where is authentication handled?"
```

Options: `--provider {huggingface,anthropic,ollama,none}`, `--model`, `-k`, `--mode {hybrid,dense,bm25}`, `--rerank`, `--rebuild`.

**Web app and terminal on the API**

Both clients can hand indexing and answering to a running API (below) instead of loading the models themselves:

```bash
python -m rag_pipeline.main --api http://127.0.0.1:8000 --path sample_codebase --provider ollama
RAG_API_URL=http://127.0.0.1:8000 streamlit run streamlit/app.py
```

The CLI registers the folder, waits for its index job and asks through `/chat` in agent mode (`--oneshot` for a single retrieval); `--rebuild` and `--mode` only apply to the local index. With `RAG_API_URL` set, the web app indexes folders and GitHub repositories through the API, shows the job's progress, and adds an **Agent mode** switch; file uploads need the in-process mode. Without `--api` or `RAG_API_URL` both work as before, which is what Streamlit Community Cloud uses. The same question gets the same sources from either client. Python code can use the client directly: `api.client.RagClient("http://127.0.0.1:8000")` has `add_repo`, `index`, `wait_for_job`, `search`, `chat_events` (the pipeline's `Step`, `Sources` and `Token` events) and `ask`.

**API**

```bash
python -m api            # http://127.0.0.1:8000, interactive docs at /docs
```

| Endpoint | What it does |
|---|---|
| `POST /repos` | Registers `{"path": "..."}` or `{"github_url": "https://github.com/owner/repo"}` and returns its id |
| `POST /repos/{id}/index` | Starts a background index job (or returns the one already running) |
| `GET /jobs/{job_id}` | Job status, stage and progress (`done` of `total`) |
| `POST /repos/{id}/search` | `{"query": "...", "k": 6}`: ranked passages with file and line numbers |
| `POST /repos/{id}/chat` | `{"question": "...", "history": [...], "mode": "agent"}`: streams `step`, `sources` and `token` events, then `done` or `error` |

By default `/chat` runs the agent: the model works through the codebase with four tools (`search_code`, `read_file`, `list_dir` and `grep`, all limited to the indexed files), so it can follow a call from one file into the next. Each tool call shows up as a step (`Searching: password hashing`, `Reading api/app.py lines 40-80`), the sources are the passages it actually opened and cited, and the answer's `[n]` citations point at them. It stops after 8 tool calls. Models that cannot call tools fall back to a single retrieval, and `"mode": "oneshot"` asks for that directly.

The API, the CLI and the web app give the same folder or GitHub repository the same index, so an index built in one is reused by the others. Registered repositories live in memory, so register them again after a restart; their indexes stay on disk.

**Index worker (Qdrant + Redis)**

By default index jobs run on threads inside the API and the index is a local Chroma folder. For large repositories, run indexing in a separate worker process instead. Chroma cannot take writes from two processes, so this mode uses Qdrant:

```bash
docker compose -f docker-compose.dev.yml up -d   # Qdrant on 127.0.0.1:6333, Redis on 127.0.0.1:6379
```

Then set these in `.env` (or the shell) and start both processes:

```bash
# RAG_VECTOR_STORE=qdrant
# REDIS_URL=redis://127.0.0.1:6379
python -m api.worker     # or: arq api.worker.WorkerSettings
python -m api
```

The API queues each index job in Redis and reads its progress back from there, so `GET /jobs/{job_id}` works as before. Both processes refuse to start in worker mode with `RAG_VECTOR_STORE=chroma`. On this laptop (RTX 4050) the worker indexed `chromadb`'s 2,987 chunks in 27 s while `/health` answered in 6.5 ms (median).

---

## ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed text model, e.g. `jinaai/jina-embeddings-v2-base-code` for large codebases |
| `RAG_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Any fastembed cross-encoder |
| `RAG_EMBEDDING_DEVICE` | `auto` | `auto` uses an NVIDIA GPU when `onnxruntime-gpu` is installed; `cpu` forces the CPU |
| `RAG_PERSIST_DIR` | `chroma_db` | Where the Chroma index is stored |
| `RAG_VECTOR_STORE` | `chroma` | `chroma` (local folder, no server) or `qdrant` (shared server, hybrid search inside Qdrant) |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant server used when `RAG_VECTOR_STORE=qdrant` |
| `REDIS_URL` | unset | Set it to send API index jobs to the worker (`python -m api.worker`); needs `RAG_VECTOR_STORE=qdrant` |
| `RAG_ALLOWED_ROOT` | unset | Restricts the folder-path input (web app and API) to one directory tree. Set this on public deployments. |
| `RAG_REPO_CACHE` | `.repos` | Where the API clones GitHub repositories |
| `RAG_API_URL` | unset | Makes the web app index and answer through a running API instead of in its own process |
| `RAG_WARM_UP` | `1` | `0` skips loading the embedding model and the Ollama model at startup |
| `HF_MODEL`, `ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT`, `OLLAMA_MODEL`, `OLLAMA_HOST` | see `.env.example` | Provider overrides |

---

## 📊 Evaluation

`evaluation/golden_set.json` holds 20 labelled questions about `sample_codebase` (natural-language questions and exact identifier lookups). The benchmark indexes the sample project and scores each retrieval strategy:

```bash
python -m evaluation.evaluate_retrieval
python -m evaluation.evaluate_retrieval --store qdrant   # same benchmark on Qdrant (--qdrant-url :memory: needs no server)
```

Results on the sample project (hit@k = an expected file is in the top k; MRR = mean reciprocal rank):

| Pipeline | hit@1 | hit@3 | MRR |
|---|---|---|---|
| Previous design: 1000-char splits, `all-MiniLM-L6-v2`, dense only | 0.75 | 0.90 | 0.839 |
| Previous splits with `bge-small-en-v1.5` | 0.85 | 1.00 | 0.917 |
| Structure-aware chunks + headers, dense only | 0.90 | 1.00 | 0.950 |
| **Structure-aware chunks + hybrid BM25/dense (default)** | **0.95** | **1.00** | **0.975** |
| Default + cross-encoder reranking | 0.95 | 1.00 | 0.975 |
| Qdrant hybrid (dense + sparse fused in one query) | 1.00 | 1.00 | 1.000 |

Reranking adds no gain on this small, already well-ranked set and costs a few seconds per question on CPU, so it is opt-in (sidebar toggle or `--rerank`). With the weaker `all-MiniLM-L6-v2` embeddings it lifts hybrid MRR from 0.938 to 0.975, so it is worth enabling for large codebases or lighter embedding models. Extend `golden_set.json` with questions about your own code to tune these choices.

**Answers: agent vs one-shot.** `evaluation/agent_questions.json` holds 10 cross-file questions about this repository and `sample_codebase`, each with the files a good answer must use. The script asks every question in both modes and scores the citations: file recall (how many of those files the answer cites) and citation validity (every `[n]` exists and points at a relevant file).

```bash
python evaluation/evaluate_answers.py                       # Ollama, OLLAMA_MODEL or the first pulled model
python evaluation/evaluate_answers.py --provider anthropic --modes agent --out results.json
```

With `qwen3.5:4b` on Ollama (RTX 4050 laptop):

| Mode | File recall | Valid citations | Seconds per question |
|---|---|---|---|
| One-shot retrieval | 0.62 | 0.63 | 17.4 |
| **Agent (default)** | **0.90** | **0.82** | **13.3** |

---

## 🧪 Tests and code style

```bash
pip install -r requirements-dev.txt
pytest              # fast offline suite (fake embeddings, fake LLMs)
pytest -m slow      # end-to-end tests with the real models, and the worker on Redis + Qdrant when they run
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
