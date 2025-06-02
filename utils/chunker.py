# utils/chunker.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

class CodebaseChunker:
    def __init__(self, codebase_dir, chunk_size=500, chunk_overlap=50):
        self.codebase_dir = codebase_dir
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_and_chunk(self):
        # ⛏️ You still need to implement logic to load .py/.md/.json files here
        # Example stub logic:
        from langchain.docstore.document import Document
        import os

        docs = []
        for root, _, files in os.walk(self.codebase_dir):
            for file in files:
                if file.endswith((".py", ".md", ".json")):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        docs.append(Document(page_content=text, metadata={"source": path}))

        return self.splitter.split_documents(docs)
