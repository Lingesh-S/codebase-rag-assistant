# rag_pipeline/test_generator.py

from generator import FlanT5Generator

# Sample context from your project
context = """
The Retriever is responsible for querying a vector database (FAISS or Chroma) using embeddings.
It returns the most relevant documents based on the input query embedding.
"""

# Sample question
question = "What is the role of the Retriever?"

# Initialize and run the generator
generator = FlanT5Generator()
answer = generator.generate(question, context)

print("\n💡 Question:", question)
print("🧠 Answer:", answer)
