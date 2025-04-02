import chromadb

# Initialize ChromaDB client
client = chromadb.Client()

# List of collections
projects = ["ProjectA", "ProjectB", "ProjectC", "General"]

# Create collections and add sample data
for project in projects:
    collection = client.create_collection(project)
    collection.add(
        documents=[f"Sample document for {project}"],
        metadatas=[{"source": "sample"}],
        ids=[f"doc1_{project}"]
    )

print("ChromaDB collections created with sample data.")
