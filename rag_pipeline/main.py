# rag_pipeline/main.py

import os
import shutil
from rag_pipeline import RAGPipeline

if __name__ == "__main__":
    codebase_path = "./sample_codebase"  # 🛠️ Change to your actual folder name

    # ✅ Step 3: Force-delete existing Chroma DB (for dev mode/testing)
    persist_dir = "chroma_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print("🧹 Old Chroma DB deleted.")

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
