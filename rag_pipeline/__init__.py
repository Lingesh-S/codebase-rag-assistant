# rag_pipeline/__init__.py

from utils.chunker import CodebaseChunker
from embeddings.embedder import get_embedder
from vectorstore.vector_store import VectorStore
from retriever.code_retriever import load_retriever
from rag_pipeline.generator import FlanT5Generator

class RAGPipeline:
    def __init__(self, codebase_dir):
        print("🔍 Initializing RAG pipeline...")

        self.chunker = CodebaseChunker(codebase_dir)
        self.embedder = get_embedder()
        self.vectorstore = VectorStore(persist_dir="chroma_db")
        self.docs = []

        self.retriever = load_retriever(
            persist_directory="chroma_db",
            embedding_function=self.embedder
        )

        self.generator = FlanT5Generator()

    def build_knowledge_base(self):
        if self.vectorstore.db_exists():
            print("⚠️ Chroma DB already exists. Skipping embedding and loading it.")
            self.vectorstore.load()
            return

        print("📄 Loading and chunking documents...")
        self.docs = self.chunker.load_and_chunk()
        print(f"✅ {len(self.docs)} chunks ready for embedding.")

        print("🧠 Generating embeddings...")
        self.embeddings = self.embedder.embed(self.docs)

        print("📦 Storing embeddings in vector DB...")
        self.vectorstore.create_from_documents(self.docs)
        print("✅ Knowledge base built successfully!")

    def answer_question(self, query):
        print(f"❓ User query: {query}")
        relevant_docs = self.retriever.retrieve(query, top_k=3)

        print("🔗 Retrieved relevant documents:")
        for doc in relevant_docs:
            print("-", doc)

        context = " ".join(relevant_docs)
        answer = self.generator.generate_answer(query, context)
        print("🧠 Answer:", answer)
        return answer
