import chromadb

# Initialize ChromaDB client
client = chromadb.Client()

# Create collections for each project and a general one
client.create_collection("ProjectA")
client.create_collection("ProjectB")
client.create_collection("ProjectC")
client.create_collection("General")

print("ChromaDB collections created.")
