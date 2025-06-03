#  Import necessary modules

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated import
from langchain_community.vectorstores import Chroma 

#  1. Define the embedding model — we'll use SentenceTransformers (lightweight, free)
def get_embedder():
    model_name = "all-MiniLM-L6-v2"  # Fast & accurate small model
    return HuggingFaceEmbeddings(model_name=model_name)

#  2. Embed chunks and store in Chroma
def embed_and_store(chunks, persist_dir="chroma_db"):
    embedder = get_embedder()

    # Create or load Chroma vector store with persistent directory
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedder, persist_directory=persist_dir)

    return vectordb
