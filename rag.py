import chromadb
import requests
import json
import re

# --- Query Functions ---
def query_shared_collection(project, query_text, n_results=2):
    """Retrieve multiple documents from the shared project collection."""
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(f"{project}_shared")
    results = collection.query(query_texts=[query_text], n_results=n_results)
    if results['documents'] and results['documents'][0]:
        return "\n".join([f"- {doc}" for doc in results['documents'][0]])
    return "No shared project data available."

def query_user_collection(user_id, project, query_text, n_results=1):
    """Retrieve documents from the user-specific collection."""
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = f"{project}_{user_id}"
    try:
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[query_text], n_results=n_results)
        if results['documents'] and results['documents'][0]:
            return "\n".join([f"- {doc}" for doc in results['documents'][0]])
        return "No user-specific data available."
    except ValueError:
        return "No user-specific data available."

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
    """Generate an optimized RAG prompt and get a response."""
    # Retrieve documents
    shared_docs = query_shared_collection(project, query_text, n_results=2)
    user_docs = query_user_collection(user_id, project, query_text, n_results=1)
    
    # Determine query type and tailor instructions
    is_summarization = bool(re.search(r"summari[zs]e|summary", query_text, re.IGNORECASE))
    if is_summarization:
        instruction = """Provide a concise summary of the video transcript, incorporating relevant details from the shared project data."""
    else:
        instruction = """Answer the query based on the provided context, using both shared and user-specific data."""

    # Build the optimized prompt
    prompt = f"""You are an assistant helping with project-related queries. You have access to two types of information:
1. **Shared Project Data**: General information about the project, accessible to all users.
2. **User-Specific Data**: Information from the user's uploaded video transcript, private to them.

{instruction} If there is no user-specific data, rely solely on the shared project data to provide a response.

**Shared Project Data**:
{shared_docs}

**User-Specific Data**:
{user_docs}

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
    query1 = "What is the main topic?"
    response1 = rag_query(user_id, project, query1)
    print("Question Response:")
    print(response1)
    
    # Test with a summarization request
    query2 = "Summarize the video"
    response2 = rag_query(user_id, project, query2)
    print("\nSummarization Response:")
    print(response2)
    
    # Test with a different user (no user-specific data yet)
    # user_id2 = "user2"
    # query3 = "What is the main topic?"
    # response3 = rag_query(user_id2, project, query3)
    # print("\nUser2 (No Transcript) Response:")
    # print(response3)