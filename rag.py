import chromadb
import requests
import json
import re
from sentence_transformers import SentenceTransformer
import os

# Load the embedding model
model_path = os.path.join(os.path.dirname(__file__), "models", "all-mpnet-base-v2")
embedding_model = SentenceTransformer(model_path)

# --- Query Functions ---

def query_shared_collection(project, query_text, n_results=1):
    """Retrieve multiple documents from the shared project collection."""
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(f"{project}_shared")
    query_embedding = embedding_model.encode(query_text).tolist()  # Encode the query
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    if results['documents'] and results['documents'][0]:
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])
    return "No shared project data available."

def query_user_collection(user_id, project, query_text, n_results=1):
    """Retrieve document and meeting type from the user-specific collection."""
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = f"{project}_{user_id}"
    try:
        collection = client.get_collection(collection_name)
        query_embedding = embedding_model.encode(query_text).tolist()  # Encode the query
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        if results['documents'] and results['documents'][0]:
            doc = results['documents'][0][0]  # Get the first document
            meeting_type = results['metadatas'][0][0].get("meeting_type", "Unknown")  # Get meeting type from metadata
            return doc, meeting_type
        return "No user-specific data available.", "Unknown"
    except ValueError:
        return "No user-specific data available.", "Unknown"

# --- Llama 3 Response ---
def get_llama3_response(prompt):
    """Call Llama 3 API with streaming response."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3:8b", "prompt": prompt, "stream": True},
        stream=True
    )
    response.raise_for_status()
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode('utf-8'))
                if 'response' in json_data:
                    full_response += json_data['response']
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON chunk: {e}")
                continue
    return full_response

# --- Optimized RAG Query ---
def rag_query(user_id, project, query_text):
    """Generate an optimized RAG prompt tailored to the meeting type and query intent."""
    # Retrieve documents and meeting type
    shared_docs = query_shared_collection(project, query_text, n_results=2)
    user_doc, meeting_type = query_user_collection(user_id, project, query_text, n_results=1)
    
    # Set base instruction based on meeting type
    if meeting_type == "Technical Meeting":
        base_instruction = "Focus on technical details, problem-solving steps, project updates, and action items."
    elif meeting_type == "KT Session":
        base_instruction = "Summarize key knowledge points, explain concepts clearly with examples if possible, and highlight processes or best practices."
    elif meeting_type == "Townhall Meeting":
        base_instruction = "Offer insights into company announcements, policy changes, and strategic directions in a concise and narrative style."
    else:
        base_instruction = "Provide a relevant response based on the available data."

    # Detect query intent
    is_summarization = bool(re.search(r"summari[zs]e|summary", query_text, re.IGNORECASE))
    is_explanation = bool(re.search(r"explain|why|how", query_text, re.IGNORECASE))
    
    if is_summarization:
        instruction = f"Provide a concise summary (2-3 sentences) that {base_instruction.lower()}"
    elif is_explanation:
        instruction = f"Offer a clear, step-by-step explanation that {base_instruction.lower()}"
    else:
        instruction = f"Answer the query directly and accurately, ensuring to {base_instruction.lower()}"

    # Build the optimized prompt
    prompt = f"""You are a professional AI assistant for an IT company, specializing in analyzing and responding to queries based on technical meetings, knowledge transfer (KT) sessions, and townhall meetings. Your goal is to deliver clear, accurate, and contextually relevant responses to assist employees.

**Instructions**:
- {instruction}
- Use a professional yet approachable tone.
- For technical queries, provide detailed explanations and define complex terms if necessary; for broader queries, focus on key points.
- If the query is unclear, ask the user for clarification.
- If user-specific data is unavailable, rely on shared project data to provide a general response.

**Shared Project Data**:
{shared_docs}

**User-Specific Data ({meeting_type} Transcript)**:
{user_doc if user_doc != "No user-specific data available." else "None available."}

**Query**:
{query_text}

**Response**:
"""
    # Get response from Llama 3
    response = get_llama3_response(prompt)
    return response

# --- Test the Optimized RAG ---
if __name__ == "__main__":
    user_id = "user1"
    project = "ProjectA"
    
    # Test with a question
    query1 = "When was Shivaji named Chhatrapati?"
    response1 = rag_query(user_id, project, query1)
    print("Question Response:")
    print(response1)
    
    # Test with a summarization request
    query2 = "Summarize the Battle of Purandar"
    response2 = rag_query(user_id, project, query2)
    print("\nSummarization Response:")
    print(response2)