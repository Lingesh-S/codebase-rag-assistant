# retriever/code_retriever.py


from langchain_chroma import Chroma  # ✅ use this instead of langchain_community
from langchain_huggingface import HuggingFaceEmbeddings

# 🔁 Load the same embedding model used before
def load_retriever(persist_directory="chroma_db"):
    model_name = "all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=model_name)

    # ♻️ Reload the persisted Chroma DB
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # 🔍 Create retriever interface from vector DB
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # return top 3 chunks
    return retriever
