import os
import shutil
import tempfile
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PythonLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Codebase RAG Assistant", layout="wide")
st.markdown("## 🧠 Codebase RAG Assistant")
st.markdown("Ask questions about your codebase. Powered by embeddings + retrieval.")

# Sidebar for input method
st.sidebar.header("📂 Codebase Configuration")
upload_option = st.sidebar.radio("Choose input method:", ("📁 Folder Path", "📤 Upload Files"))

# File upload logic
uploaded_files = []
folder_path = None

if upload_option == "📁 Folder Path":
    folder_path = st.sidebar.text_input("Enter path to your codebase folder:", value="sample_codebase")
else:
    uploaded_files = st.sidebar.file_uploader("Upload your codebase files", type=["py", "md", "json"], accept_multiple_files=True)

# Button to build knowledge base
if st.sidebar.button("📦 Build Knowledge Base"):
    st.session_state.docs = []
    st.session_state.vectordb = None

    if upload_option == "📤 Upload Files" and uploaded_files:
        temp_dir = tempfile.mkdtemp()
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        folder_path = temp_dir

    if folder_path and os.path.exists(folder_path):
        loaders = [
            DirectoryLoader(path=folder_path, glob="**/*.py", loader_cls=PythonLoader),
            DirectoryLoader(path=folder_path, glob="**/*.md", loader_cls=TextLoader),
            DirectoryLoader(path=folder_path, glob="**/*.json", loader_cls=JSONLoader),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
        vectordb.persist()

        st.session_state.vectordb = vectordb
        st.success("✅ Knowledge base built successfully!")
    else:
        st.warning("❗ Please upload files or enter a valid folder path.")

# Ask a question
query = st.text_input("❓ Ask a question about your codebase", placeholder="Your question...")
if st.button("🔍 Search") and query:
    if "vectordb" in st.session_state and st.session_state.vectordb:
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.2, "max_length": 200})
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.vectordb.as_retriever())
        response = qa.run(query)
        st.markdown("### 🧾 Answer")
        st.success(response)
    else:
        st.warning("⚠️ Please build the knowledge base first.")
