# 🧠 Codebase RAG Assistant — AI-Powered Code Explainer
An advanced Retriever-Augmented Generation (RAG) pipeline designed to ingest, index, and query software codebases using large language models.
Explore your own project files with natural language queries — understand, debug, and document your code effortlessly.


---
## 🚧 Development Notice
![Project Status](https://img.shields.io/badge/status-in--progress-yellow)

This project is currently under active development.  
📅 Expected updates coming soon!

---

## 🎯 Objective

- Loads and chunks various code files (`.py`, `.md`, `.json`, etc.)
- Creates semantic vector embeddings for efficient search
- Retrieves relevant code snippets in response to natural language queries
- Uses LLMs to generate detailed, context-aware explanations
- Provides a developer-friendly UI for interactive Q&A on your codebase

---


##🔧 Tech Stack
- LangChain — Orchestrating retriever and generator workflows
- SentenceTransformers — Embedding models for semantic search
- ChromaDB — Efficient vector similarity search and indexing
- Streamlit — Web UI for interactive code queries
- Python — Core programming language



---


## 📂 Project Structure

```bash
CODEBASE-RAG-ASSISTANT/
├── .venv/               # Python virtual environment folder
├── app/                 # Streamlit or application UI code
├── chroma_db/           # Chroma vector DB persistence files
├── data/                # Project data, documents, or knowledge base
├── embeddings/          # Embedding generation scripts and models
├── loaders/             # Code loaders for .py, .md, .json, etc.
├── rag_pipeline/        # Orchestration of retriever + generator
├── retriever/           # Semantic search and vector similarity logic
├── sample_codebase/     # Example codebases for demo/testing
├── utils/               # Utility functions and helpers
├── vectorstore/         # Vector store management and setup
├── venv/                # Another possible Python virtual environment folder
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🔍 Key Features

- 📂 Multi-format code ingestion (.py, .md, .json)
- 🧠 Semantic search over code chunks using embeddings
- 🤖 LLM-powered natural language answers explaining code
- 🔄 Modular components for easy customization and extension
- 🛠️ Developer-friendly with config options for vector stores and LLMs


---

## 💻 Installation

```bash
git clone https://github.com/your-username/codebase-rag-assistant.git
cd codebase-rag-assistant
pip install -r requirements.txt

```

---

## ▶️ Running the Pipeline (Planned)

```bash
# Example placeholder - to be updated when modules are ready
python pipeline/main.py
```

---

## ▶️ How to Use
1. Prepare your codebase in the data/ folder or point to your own repo

2. Run the loader to ingest and chunk code files into the vector store

3. Start the Streamlit app for an interactive Q&A interface

---

## 📜 License

MIT License © 2025 Lingesh S
