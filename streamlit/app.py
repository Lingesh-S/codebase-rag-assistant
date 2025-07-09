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

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🧠 Codebase RAG Assistant", layout="wide")
st.title("💻 Codebase RAG Assistant")
st.caption("🚀 Ask intelligent questions about your codebase using a Retrieval-Augmented Generation (RAG) pipeline.")

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")
upload_option = st.sidebar.radio("Select input mode:", ["📁 Folder Path", "📤 Upload Files"])

uploaded_files = []
folder_path = None

if upload_option == "📁 Folder Path":
    folder_path = st.sidebar.text_input("Enter your local folder path", value="sample_codebase")
else:
    uploaded_files = st.sidebar.file_uploader("Upload .py / .md / .json files", type=["py", "md", "json"], accept_multiple_files=True)

# --- Build Knowledge Base Button ---
if st.sidebar.button("📦 Build Knowledge Base"):
    st.session_state.docs = []
    st.session_state.vectordb = None

    # Handle uploaded files (write to temp dir)
    if upload_option == "📤 Upload Files" and uploaded_files:
        temp_dir = tempfile.mkdtemp()
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        folder_path = temp_dir

    # Load and index files
    if folder_path and os.path.exists(folder_path):
        with st.spinner("🔍 Processing files and generating embeddings..."):
            loaders = [
                DirectoryLoader(folder_path, glob="**/*.py", loader_cls=PythonLoader),
                DirectoryLoader(folder_path, glob="**/*.md", loader_cls=TextLoader),
                DirectoryLoader(folder_path, glob="**/*.json", loader_cls=JSONLoader),
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
        st.warning("⚠️ Please upload files or provide a valid folder path.")

# --- Question Input Section ---
st.markdown("---")
st.subheader("❓ Ask Your Question")

query = st.text_input("What do you want to know about your code?", placeholder="e.g., What does the User class do?")

if st.button("🔍 Get Answer"):
    if "vectordb" in st.session_state and st.session_state.vectordb:
        with st.spinner("🧠 Thinking..."):
            llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.2, "max_length": 200})
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.vectordb.as_retriever())
            response = qa.run(query)

        st.markdown("### 🧾 Answer")
        with st.expander("🧠 View Answer"):
            st.code(response, language="markdown")
    else:
        st.warning("⚠️ Please build the knowledge base first.")
