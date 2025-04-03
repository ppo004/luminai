import chromadb

# Set up persistent ChromaDB client
persist_directory = "chroma_db"
client = chromadb.PersistentClient(path=persist_directory)

# Define projects and their shared data files
projects = {
    "ProjectA": "transcripts/shivaji.txt"
}

# Seed shared collections
for project, file_path in projects.items():
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Create or get the shared collection
    collection_name = f"{project}_shared"
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        documents=[data],
        metadatas=[{"source": file_path}],
        ids=[f"shared_doc_{project}"]
    )

print("Shared ChromaDB collections seeded.")
