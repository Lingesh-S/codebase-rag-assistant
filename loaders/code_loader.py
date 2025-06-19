# loaders/code_loader.py
from langchain_community.document_loaders import DirectoryLoader, PythonLoader, TextLoader, JSONLoader


def load_codebase(directory_path):
    # Load .py files using PythonLoader
    py_loader = DirectoryLoader(path=directory_path, glob="**/*.py", loader_cls=PythonLoader)

    # Optionally, load .md files using TextLoader
    md_loader = DirectoryLoader(path=directory_path, glob="**/*.md", loader_cls=TextLoader)

    # Optionally, load .json files using JSONLoader
    json_loader = DirectoryLoader(path=directory_path, glob="**/*.json", loader_cls=JSONLoader)

    # Load all the documents
    documents = py_loader.load() + md_loader.load() + json_loader.load()
    return documents
  
