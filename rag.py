import chromadb
import requests
import json
import re
from utils import embedding_model


# --- Query Functions ---

def query_shared_collection(project, query_text, n_results=2):
    """Retrieve multiple documents from the shared project collection."""
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(f"{project}_shared")
        query_embedding = embedding_model.encode(query_text).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        if results['documents'] and results['documents'][0]:
            return "\n".join([f"- {doc}" for doc in results['documents'][0]])
        return "No shared project data available."
    except ValueError:
        return "Shared project data not initialized."

def query_user_collection(user_id, project, query_text, n_results=1):
    """Retrieve document and meeting type from the user-specific collection."""
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = f"{project}_{user_id}"
    try:
        collection = client.get_collection(collection_name)
        query_embedding = embedding_model.encode(query_text).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        if results['documents'] and results['documents'][0]:
            doc = results['documents'][0][0]
            meeting_type = results['metadatas'][0][0].get("meeting_type", "Unknown")
            return doc, meeting_type
        return "No user-specific data available.", "Unknown"
    except ValueError:
        return "No user-specific data available.", "Unknown"

# --- Llama 3 Response ---
def get_llama3_response(prompt):
    """Call Llama 3 API with streaming response."""
    try:
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
        return full_response.strip() or "No response generated."
    except requests.RequestException as e:
        return f"Error contacting Llama 3 API: {str(e)}"

# --- Enhanced RAG Query ---
def rag_query(user_id, project, query_text):
    """Generate an optimized RAG prompt with improved intent detection and formatting."""
    # Retrieve documents and meeting type
    shared_docs = query_shared_collection(project, query_text, n_results=2)
    user_doc, meeting_type = query_user_collection(user_id, project, query_text, n_results=1)
    
    # Enhanced Query Intent Detection
    query_lower = query_text.lower()
    is_summarization = bool(re.search(r"summari[zs]e|summary|overview", query_lower))
    is_explanation = bool(re.search(r"explain|how|why|what is", query_lower))
    is_list = bool(re.search(r"list|steps|items|details", query_lower))
    
    # Meeting Type Instructions
    meeting_instructions = {
        "Technical Meeting": "Focus on technical details, problem-solving steps, and action items.",
        "KT Session": "Explain concepts clearly with examples, focusing on processes or best practices.",
        "Townhall Meeting": "Provide concise insights into announcements, policies, or strategic directions.",
        "Unknown": "Respond based on available data, keeping it relevant and clear."
    }
    base_instruction = meeting_instructions.get(meeting_type, meeting_instructions["Unknown"])

    # Build Instruction Based on Intent
    if is_summarization:
        instruction = f"Summarize in 2-3 sentences, {base_instruction.lower()}"
        format_instruction = "Use plain text."
    elif is_explanation:
        instruction = f"Explain step-by-step, {base_instruction.lower()}"
        format_instruction = "Use numbered steps if applicable."
    elif is_list:
        instruction = f"Provide a detailed response, {base_instruction.lower()}"
        format_instruction = "Use bullet points for key items or steps."
    else:
        instruction = f"Answer directly, {base_instruction.lower()}"
        format_instruction = "Use plain text; if the query is unclear, ask for clarification."

    # Optimized Prompt
    prompt = f"""You are an IT assistant for technical meetings, KT sessions, and townhalls.  
**Instructions**:  
- {instruction}  
- {format_instruction}  
- Keep responses professional, concise, and relevant. Define technical terms if needed.  
**Shared Data**:  
{shared_docs}  
**User Data ({meeting_type})**:  
{user_doc}  
**Query**:  
{query_text}  
**Response**:  
"""

    # Get and return response
    response = get_llama3_response(prompt)
    return response

# Example usage
# if __name__ == "__main__":
#     print(rag_query("user1", "ProjectA", "Summarize the last technical meeting"))
#     print(rag_query("user1", "ProjectA", "How does the deployment process work?"))
#     print(rag_query("user1", "ProjectA", "List the action items from the KT session"))