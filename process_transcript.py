import chromadb
from sentence_transformers import SentenceTransformer

def process_transcript(user_id, project, transcript_path):
    # Set up persistent ChromaDB client
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Create or get user-specific collection
    collection_name = f"{project}_{user_id}"
    collection = client.get_or_create_collection(collection_name)
    
    # Load embedding model
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Read and process transcript
    with open(transcript_path, 'r', encoding='utf-8') as file:
        transcript = file.read()
    
    # embedding = model.encode(transcript).tolist()
    collection.add(
        documents=[transcript],
        # embeddings=[embedding],
        metadatas=[{"source": transcript_path}],
        ids=[f"transcript_{user_id}"]
    )
    
    print(f"Transcript for {user_id} in {project} processed.")

# Example usage
if __name__ == "__main__":
    process_transcript("user1", "ProjectA", "transcripts/transcript.txt")
