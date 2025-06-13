# vectorstore/vector_store.py
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def split_documents(documents: list[Document], chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def db_exists(persist_directory: str) -> bool:
    return os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0

def create_or_load_vectorstore(
    docs: list[Document],
    embedding_model: Embeddings,
    persist_directory: str = "vectorstore"
):
    if db_exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

def similarity_search(vectorstore: Chroma, query: str, k: int = 5):
    return vectorstore.similarity_search(query, k=k)

# ✅ Restored VectorStore class to support class-based usage in rag_pipeline

class VectorStore:
    def __init__(self, persist_dir="vectorstore", embedding=None):
        self.persist_dir = persist_dir
        self.embedding = embedding
        self.db = None

    def create_from_documents(self, docs: list[Document]):
        self.db = Chroma.from_documents(documents=docs, embedding=self.embedding, persist_directory=self.persist_dir)
        return self.db

    def load(self):
        self.db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding)
        return self.db

    def similarity_search(self, query, k=5):
        if not self.db:
            self.load()
        return self.db.similarity_search(query, k=k)

    def db_exists(self):
        return os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0
