# rag_pipeline/__init__.py

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
