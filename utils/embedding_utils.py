"""
Embedding utilities for vector operations.
"""
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import config

def get_embedding_model():
    """
    Get the embedding model instance.
    
    Returns:
        OllamaEmbeddings: Configured embedding model
    """
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL, 
        base_url=config.OLLAMA_BASE_URL
    )

def get_vectorstore(project, collection_suffix="shared"):
    """
    Get a vector store for a specific project.
    
    Args:
        project (str): Project identifier
        collection_suffix (str): Collection suffix (default: "shared")
        
    Returns:
        Chroma: Configured vector store
    """
    embedding_model = get_embedding_model()
    collection_name = f"{project}_{collection_suffix}"
    
    return Chroma(
        collection_name=collection_name,
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

def format_docs(docs):
    """
    Helper function to format retrieved documents into a string.
    
    Args:
        docs (list): List of documents
        
    Returns:
        str: Formatted document string
    """
    return "\n".join([f"- {doc.page_content}" for doc in docs])
