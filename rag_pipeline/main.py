# rag_pipeline/main.py

from rag_pipeline import RAGPipeline  # ✅ Import the class from __init__.py

if __name__ == "__main__":
    # Set your codebase path
    codebase_path = "./sample_codebase"  # Update to your actual codebase folder
    query = "What does the main function do?"

    rag = RAGPipeline(codebase_path)
    rag.build_knowledge_base()
    rag.answer_question(query)
