import chromadb
from sentence_transformers import SentenceTransformer

def process_transcript(project, transcript_path):
    # Initialize ChromaDB client
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)

    # Load the embedding model, specifying from_tf=True
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Read the transcript
    with open(transcript_path, 'r') as file:
        transcript = file.read()

    # Generate embedding
    embedding = model.encode(transcript).tolist()

    # Get or create the collection for the project
    collection = client.get_or_create_collection(project)

    # Add the transcript to the collection
    collection.add(
        documents=[transcript],
        embeddings=[embedding],
        metadatas=[{"source": transcript_path}],
        ids=[f"transcript_{project}"]
    )

    print(f"Transcript for {project} processed and added to the collection.")

# Example usage
if __name__ == "__main__":
    project = "ProjectA"  # Assume user selected Project A
    transcript_path = "transcripts/shivaji.txt"
    process_transcript(project, transcript_path)