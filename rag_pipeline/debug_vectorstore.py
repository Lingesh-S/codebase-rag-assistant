
# rag_pipeline/debug_vectorstore.py

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import

# Load the existing vectorstore
vectorstore = Chroma(
    persist_directory="chroma_db",  # ✅ Correct directory
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Search for any relevant chunk
query = "greet"
docs = vectorstore.similarity_search(query, k=3)

print(f"🔍 Found {len(docs)} matching docs:")
for doc in docs:
    print("🧩 Chunk:\n", doc.page_content[:300])
    print("="*50)
