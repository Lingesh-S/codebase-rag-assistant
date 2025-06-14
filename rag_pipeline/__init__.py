# rag_pipeline/__init__.py

# ✅ LangChain v0.2+ Compatible Imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    NotebookLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

class RAGPipeline:
    def __init__(self, codebase_path, persist_directory="chroma_db"):
        self.codebase_path = codebase_path
        self.persist_directory = persist_directory

        # ✅ New embedding model setup
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_knowledge_base(self):
        # ✅ 1. Load supported files
        file_paths = []
        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith(('.py', '.md', '.json', '.ipynb')):
                    file_paths.append(full_path)

        # ✅ 2. Use loaders based on file extension
        documents = []
        for path in file_paths:
            try:
                if path.endswith(".py"):
                    loader = TextLoader(path)
                elif path.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(path)
                elif path.endswith(".json"):
                    loader = JSONLoader(path, jq_schema=".")
                elif path.endswith(".ipynb"):
                    loader = NotebookLoader(path)
                else:
                    continue
                documents.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")

        print(f"📄 Loaded {len(documents)} documents.")

        # ✅ 3. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        print(f"🧩 Split into {len(texts)} chunks.")

        if not texts:
            print("⚠️ No valid chunks found after splitting. Please check your documents.")
            return

        # ✅ 4. Initialize Chroma DB
        self.db = Chroma(
            collection_name="code_chunks",
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )

        if len(self.db.get()["ids"]) == 0:
            print("📚 No existing vectors found. Adding new documents...")
            self.db.add_documents(texts)
            print("✅ Knowledge base built and persisted.")
        else:
            print("📦 Using existing Chroma DB from disk.")

    def answer_question(self, query):
        retriever = self.db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        print("\n📄 Top relevant code snippets:\n")
        for i, doc in enumerate(docs, 1):
            print(f"[{i}] {doc.metadata.get('source', 'unknown')}")
            print(doc.page_content[:500])
            print("-" * 60)
