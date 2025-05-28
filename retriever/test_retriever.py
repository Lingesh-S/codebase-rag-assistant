


# retriever/test_retriever.py

from retriever.code_retriever import load_retriever

# ✅ Load the retriever
retriever = load_retriever()

# ✅ Ask a sample question
query = "What does the hello.py file do?"

# ✅ Get relevant documents
results = retriever.invoke(query)

# ✅ Print the top result(s)
for i, doc in enumerate(results):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content)
