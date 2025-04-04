
import chromadb
from sentence_transformers import SentenceTransformer
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_transcript(user_id, project, transcript_path):
    # Set up persistent ChromaDB client
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or get user-specific collection
    collection_name = f"{project}_{user_id}"
    collection = client.get_or_create_collection(collection_name)

    # Load embedding model
    model_path = os.path.join(os.path.dirname(__file__), "models", "all-mpnet-base-v2")
    model = SentenceTransformer(model_path)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Read and process transcript
    with open(transcript_path, 'r', encoding='utf-8') as file:
        transcript = file.read()

    # Split the transcript into chunks
    chunks = text_splitter.split_text(transcript)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    ids = [f"transcript_chunk_{user_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": transcript_path} for _ in range(len(chunks))]
    documents = chunks

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Transcript for {user_id} in {project} processed into {len(chunks)} chunks.")

# Example usage
if __name__ == "__main__":
    process_transcript("user1", "ProjectA", "transcripts/transcript.txt")