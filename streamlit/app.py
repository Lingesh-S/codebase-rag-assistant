# streamlit/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from rag_pipeline import RAGPipeline

# Set page config
st.set_page_config(page_title="Codebase RAG Assistant", page_icon="🤖", layout="wide")

st.title("🧠 Codebase RAG Assistant")
st.caption("Ask questions about your codebase. Powered by embeddings + retrieval.")

# --- Sidebar ---
st.sidebar.header("📂 Codebase Configuration")

# Folder input
code_folder = st.sidebar.text_input("Enter path to your codebase folder:", "sample_codebase")

# Build button
if st.sidebar.button("📦 Build Knowledge Base"):
    if os.path.exists(code_folder):
        st.session_state.pipeline = RAGPipeline(code_folder)
        with st.spinner("Building vector database..."):
            st.session_state.pipeline.build_knowledge_base()
        st.success("✅ Knowledge base built successfully!")
    else:
        st.error("❌ Folder path not found. Please check again.")

# --- Main UI ---
st.markdown("### ❓ Ask a question about your codebase")

question = st.text_input("Your question:")

if st.button("🔍 Search"):
    if "pipeline" not in st.session_state:
        st.warning("⚠️ Please build the knowledge base first.")
    elif question.strip() == "":
        st.warning("⚠️ Question cannot be empty.")
    else:
        with st.spinner("Searching relevant code chunks..."):
            results = st.session_state.pipeline.answer_question(question)

            if results:
                st.markdown("### 📄 Top Relevant Snippets")
                for idx, doc in enumerate(results, 1):
                    st.markdown(f"**[{idx}] {doc.metadata.get('source', 'Unknown File')}**")
                    st.code(doc.page_content[:1000], language='python')
            else:
                st.info("🤔 No relevant code found.")
