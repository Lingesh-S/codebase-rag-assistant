from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os 

class VectorStore:
    def __init__(self, persist_dir="chroma_db"):
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = None

    def create_from_documents(self, docs):
        self.db = Chroma.from_documents(documents=docs, embedding=self.embedding, persist_directory=self.persist_dir)
        return self.db

    def load(self):
        self.db = Chroma(persist_directory=self.persist_dir, embedding=self.embedding)
        return self.db

    def similarity_search(self, query, k=5):
        if not self.db:
            self.load()
        return self.db.similarity_search(query, k=k)

    # ✅ Add this method to match usage in main pipeline
    def store(self, docs, embedder=None):
        if embedder:
            self.embedding = embedder  # override default
        self.db = Chroma.from_documents(documents=docs, embedding=self.embedding, persist_directory=self.persist_dir)
        print("✅ Documents embedded and stored in Chroma DB.")
