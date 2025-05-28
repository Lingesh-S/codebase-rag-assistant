# loaders/test_loader.py

# 📦 Import all modules
from loaders.code_loader import load_codebase             # ✅ Import loader
from utils.chunker import split_documents                 # ✅ Chunking logic
from embeddings.embedder import embed_and_store           # ✅ Embed & store

# ✅ 1. Load raw documents from 'data/' folder now
docs = load_codebase("data")

# ✅ 2. Chunk documents
chunks = split_documents(docs)

# ✅ 3. Preview
print(f"Loaded {len(docs)} documents")
print(f"Total chunks created: {len(chunks)}")
print("Preview of first chunk:\n", chunks[0].page_content[:300])

# ✅ 4. Embed & Store in vector DB
vectordb = embed_and_store(chunks)
print("✅ Chunks embedded and stored in vector DB successfully!")
