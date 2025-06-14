# embeddings/embedder.py

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns a HuggingFaceEmbeddings instance using the specified model name.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def embed_and_store(
    chunks: list[Document],
    persist_dir: str = "chroma_db",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Embeds documents and stores them in Chroma DB.
    """
    # ✅ Debug: Preview first few chunks
    for i, doc in enumerate(chunks[:2]):
        print(f"\n📄 Chunk {i}:\n{doc.page_content[:300]}\n")

    embedder = get_embedding_model(model_name)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedder, persist_directory=persist_dir)
    print(f"✅ Stored {len(chunks)} chunks to Chroma DB at '{persist_dir}'")
    return vectordb
