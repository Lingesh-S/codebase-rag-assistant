#streamlit/app.py
import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PythonLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Codebase RAG Assistant", layout="wide", page_icon="🤖")

# --- Header ---
# --- Custom CSS Styling ---
st.markdown("""
    <style>
        /* GitHub button fixed to top right */
        .github-float {
            position: fixed;
            top: 20px;
            right: 25px;
            z-index: 999;
        }

        .github-btn {
            background-color: #00f9ff;
            color: black;
            font-weight: bold;
            padding: 6px 16px;
            border-radius: 8px;
            text-decoration: none;
            box-shadow: 0 0 10px rgba(0, 249, 255, 0.7);
        }

        .github-btn:hover {
            background-color: #00c2cc;
            color: white;
        }

        /* Heading box */
        .header-box {
            background-color: #222;
            border: 2px solid #00f9ff;
            border-radius: 15px;
            padding: 30px 20px;
            margin-bottom: 30px;
            box-shadow: 0 0 18px rgba(0, 249, 255, 0.4);
            text-align: center;
        }

        /* Toggle style */
        .stToggleSwitch span {
            color: #fff !important;
        }
    </style>

    <!-- GitHub Button -->
    <div class='github-float'>
        <a class='github-btn' href='https://github.com/Lingesh-S/codebase-rag-assistant' target='_blank'>⭐ GitHub</a>
    </div>

    <!-- Page Title Box -->
    <div class='header-box'>
        <h1 style='color: #00f9ff;'>🤖 Codebase RAG Assistant</h1>
        <p style='color: #ccc;'>Ask questions about your codebase using AI-powered document retrieval & understanding</p>
    </div>
""", unsafe_allow_html=True)



# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # 🌙 Dark Mode Toggle (for future theming use)
    st.toggle("🌙 Dark mode")

    # 🧠 Model Config
    model_name = st.selectbox("Choose LLM model", options=[
        "google/flan-t5-xl",
        "tiiuae/falcon-7b-instruct",
        "bigscience/bloom"
    ], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.slider("Max tokens", 50, 512, 200)

    st.markdown("---")
    st.header("📂 Codebase Input")

    upload_method = st.radio("Choose input type:", ["📁 Folder Path", "📤 Upload Files"])
    uploaded_files = []
    folder_path = None

    if upload_method == "📁 Folder Path":
        folder_path = st.text_input("Enter folder path:", value="sample_codebase")
    else:
        uploaded_files = st.file_uploader("Upload .py / .md / .json files", type=["py", "md", "json"], accept_multiple_files=True)

    # 🔄 Reset Chat
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    # 📦 Build Knowledge Base
    if st.button("📦 Build Knowledge Base"):
        st.session_state.docs = []
        st.session_state.vectordb = None

        if upload_method == "📤 Upload Files" and uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            folder_path = temp_dir

        if folder_path and os.path.exists(folder_path):
            with st.spinner("🔄 Loading & embedding documents..."):
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

            # 📄 Show uploaded files
            if uploaded_files:
                st.markdown("### 📄 Uploaded Files")
                for file in uploaded_files:
                    st.write(file.name)
        else:
            st.warning("⚠️ Invalid folder or no files uploaded.")

    st.markdown("---")
    st.caption("💡 Tip: Upload files or use a local folder before asking questions.")
    st.markdown("🔗 [GitHub Repo](https://github.com/Lingesh-S/codebase-rag-assistant)")

# --- Chat Section ---
st.markdown("### 💬 Ask a question about your codebase")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask your question here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if "vectordb" in st.session_state and st.session_state.vectordb:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-xl",
                    model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
                    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
                )
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.vectordb.as_retriever())
                response = qa.run(query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("⚠️ Please build the knowledge base first.")
