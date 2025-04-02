import chromadb

# Configure persistent storage
persist_directory = "chroma_db"  # Choose a directory to save the database
client = chromadb.PersistentClient(path=persist_directory)

# List of collections
projects = ["ProjectA", "ProjectB", "ProjectC", "General"]

# Create collections and add sample data
for project in projects:
    collection = client.get_or_create_collection(project) # Use get_or_create
    collection.add(
        documents=[f"Sample document for {project}"],
        metadatas=[{"source": "sample"}],
        ids=[f"doc1_{project}"]
    )

print("ChromaDB collections created (or accessed) with sample data in:", persist_directory)