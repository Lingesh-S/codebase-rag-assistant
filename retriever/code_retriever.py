# retriever/code_retriever.py

from langchain_chroma import Chroma

# 💡 Accept embedding_function from outside
def load_retriever(persist_directory="chroma_db", embedding_function=None):
    if embedding_function is None:
        raise ValueError("❌ 'embedding_function' is required!")

    # ♻️ Reload the persisted Chroma DB
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    # 🔍 Create retriever interface from vector DB
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # return top 3 chunks
    return retriever
