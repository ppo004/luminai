"""
Transcript processing utilities.
"""
import chromadb
from sentence_transformers import SentenceTransformer
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

def process_transcript(user_id, project, transcript_path, meeting_type):
    """
    Process a transcript file and add it to the vector database.
    
    Args:
        user_id (str): User identifier
        project (str): Project identifier
        transcript_path (str): Path to the transcript file
        meeting_type (str): Type of meeting (e.g., "standup", "refinement")
    """
    # Set up persistent ChromaDB client
    client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)

    # Create or get user-specific collection
    collection_name = f"{project}_{user_id}"
    collection = client.get_or_create_collection(collection_name)

    # Load embedding model
    model_path = os.path.join(config.MODELS_DIRECTORY, "all-mpnet-base-v2")
    model = SentenceTransformer(model_path)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Read and process transcript
    with open(transcript_path, 'r', encoding='utf-8') as file:
        transcript = file.read()

    # Split the transcript into chunks
    chunks = text_splitter.split_text(transcript)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    # Generate metadata and IDs
    transcript_id = os.path.basename(transcript_path).split('.')[0]
    metadatas = [{
        "source": transcript_path,
        "meeting_type": meeting_type,
        "user_id": user_id
    } for _ in range(len(chunks))]
    ids = [f"{transcript_id}_{i}" for i in range(len(chunks))]

    # Add chunks to the collection
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Processed transcript with {len(chunks)} chunks for user {user_id}, project {project}")
    return len(chunks)
