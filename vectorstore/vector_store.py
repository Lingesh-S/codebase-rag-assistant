# vector_store.py
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def split_documents(documents: list[Document], chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def db_exists(persist_directory: str) -> bool:
    return os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0

def create_or_load_vectorstore(
    docs: list[Document],
    embedding_model: Embeddings,
    persist_directory: str = "vectorstore"
):
    if db_exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

def similarity_search(vectorstore: Chroma, query: str, k: int = 5):
    return vectorstore.similarity_search(query, k=k)
