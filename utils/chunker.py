from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits a list of LangChain documents into smaller chunks.
    
    Args:
        documents (List[Document]): Input documents
        chunk_size (int): Max characters per chunk
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        List[Document]: Chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)
