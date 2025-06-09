# embedder.py

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns a HuggingFaceEmbeddings instance using the specified model name.

    Parameters:
    - model_name (str): Name of the Hugging Face model to use.

    Returns:
    - HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def embed_and_store(
    chunks: list[Document],
    persist_dir: str = "chroma_db",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Embeds documents and stores them in Chroma DB.

    Parameters:
    - chunks (list[Document]): The list of chunked documents to embed.
    - persist_dir (str): The directory to store the Chroma DB.
    - model_name (str): Embedding model to use.

    Returns:
    - Chroma vector store object.
    """
    embedder = get_embedding_model(model_name)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedder, persist_directory=persist_dir)
    print(f"✅ Stored {len(chunks)} chunks to Chroma DB at '{persist_dir}'")
    return vectordb
