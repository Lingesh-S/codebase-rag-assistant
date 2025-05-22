# retriever/loader.py

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents_from_folder(folder_path):
    """
    Load documents from a folder containing .pdf and .txt files.
    Returns a list of LangChain documents.
    """
    documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue  # skip unsupported files

        docs = loader.load()
        documents.extend(docs)

    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into chunks suitable for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
