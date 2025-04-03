import chromadb
from sentence_transformers import SentenceTransformer

persist_directory = "chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
projects = {
    "ProjectA": "transcripts/shivaji.txt"
}

for project, file_path in projects.items():
    with open(file_path, 'r', encoding='utf-8') as file:  # Add encoding
        data = file.read()
    
    collection = client.get_or_create_collection(project)
    embeddings = embedding_model.encode([data])
    collection.add(
        documents=[data],
        embeddings=embeddings,  # Add the generated embeddings
        metadatas=[{"source": file_path}],
        ids=[f"doc1_{project}"]
    )

print("ChromaDB collections seeded with project-specific data.")