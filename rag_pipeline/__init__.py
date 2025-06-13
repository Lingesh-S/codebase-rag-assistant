# rag_pipeline/__init__.py

from utils.chunker import CodebaseChunker
from embeddings.embedder import get_embedding_model
from vectorstore.vector_store import VectorStore
from retriever.code_retriever import load_retriever
from rag_pipeline.generator import FlanT5Generator


class RAGPipeline:
    def __init__(self, codebase_dir):
        print("🔍 Initializing RAG pipeline...")

        self.chunker = CodebaseChunker(codebase_dir)
        self.embedder = get_embedding_model()
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

        # ✅ Retrieve relevant docs
        relevant_docs = self.retriever.invoke(query)

        print("🔗 Retrieved relevant documents:")
        for doc in relevant_docs:
            print("-", doc.page_content)

        # ✅ Build context
        context = " ".join([doc.page_content for doc in relevant_docs])

        # 🔍 Debug: print context before passing to the model
        print("🧾 Context being sent to the model:")
        print(context)

        # 🧠 Generate answer
        answer = self.generator.generate(question=query, context=context)
        print("🧠 Answer:", answer)
        return answer
