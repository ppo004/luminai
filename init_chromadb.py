import chromadb
from sentence_transformers import SentenceTransformer
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import embedding_model as model

# Set up persistent ChromaDB client
persist_directory = "chroma_db"
client = chromadb.PersistentClient(path=persist_directory)

# Define projects and their shared data files
projects = {
    "ProjectA": "transcripts/masterData.txt"
}
model_path = os.path.join(os.path.dirname(__file__), "models", "all-mpnet-base-v2")
model = SentenceTransformer(model_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# Seed shared collections
for project, file_path in projects.items():
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    chunks = text_splitter.split_text(data)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    # Create or get the shared collection
    collection_name = f"{project}_shared"
    collection = client.get_or_create_collection(collection_name)
    ids = [f"shared_chunk_{project}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_path} for _ in range(len(chunks))]
    documents = chunks
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

print("Shared ChromaDB collections seeded.")
