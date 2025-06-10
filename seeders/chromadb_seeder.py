"""
ChromaDB initialization and seeding.
"""
import chromadb
from sentence_transformers import SentenceTransformer
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

def init_chromadb():
    """
    Initialize and seed ChromaDB with project data.
    """
    # Set up persistent ChromaDB client
    client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)

    # Define projects and their shared data files
    projects = {
        "SonarQube": os.path.join(config.DATA_FOLDER, "sonarqube1.txt")
    }
    
    # Load embedding model
    model_path = os.path.join(config.MODELS_DIRECTORY, "all-mpnet-base-v2")
    model = SentenceTransformer(model_path)
    
    # Configure text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # Seed shared collections
    for project, file_path in projects.items():
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping...")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
            
        chunks = text_splitter.split_text(data)
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()
        
        # Create or get the shared collection
        collection_name = f"{project}_shared"
        collection = client.get_or_create_collection(collection_name)
        
        # Prepare data for insertion
        ids = [f"shared_chunk_{project}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path} for _ in range(len(chunks))]
        
        # Add data to collection
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Seeded {len(chunks)} chunks for project: {project}")
    
    print("ChromaDB initialization complete")

if __name__ == "__main__":
    init_chromadb()
