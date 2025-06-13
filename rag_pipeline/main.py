# rag_pipeline/main.py

from rag_pipeline import RAGPipeline

if __name__ == "__main__":
    codebase_path = "./sample_codebase"  # 🛠️ Change to your actual folder name

    print("🔍 Initializing RAG pipeline...")
    rag = RAGPipeline(codebase_path)
    rag.build_knowledge_base()

    print("\n💬 Ask questions about your codebase! Type 'exit' to quit.\n")

    while True:
        user_input = input("❓ Ask a question about your codebase: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting assistant. See you soon!")
            break
        rag.answer_question(user_input)
