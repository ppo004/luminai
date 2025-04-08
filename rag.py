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

# --- Robust Intent Detection with Domain-Specific Keywords ---
def detect_intent(query_text):
    """Detect query intent using weighted keywords with domain-specific terms."""
    query_lower = " ".join(query_text.lower().split())  # Normalize whitespace
    words = query_lower.split()
    
    # Define intents with weighted keywords, including domain-specific terms
    intent_keywords = {
        "summarization": {
            # General
            "summarize": 4, "summary": 4, "overview": 3, "brief": 3, "give me a": 2, "short": 2, "recap": 2,
            # Domain-specific
            "status": 2, "progress": 2, "update": 2, "meeting": 1, "project": 1
        },
        "explanation": {
            # General
            "explain": 4, "how": 4, "why": 4, "what is": 3, "break down": 3, "describe": 2, "tell me": 2, "clarify": 2,
            # Domain-specific
            "works": 3, "process": 3, "deploy": 2, "authentication": 2, "jwt": 2, "database": 2, "api": 2, 
            "microservices": 2, "integration": 1, "setup": 1, "function": 1
        },
        "list": {
            # General
            "list": 4, "steps": 4, "items": 3, "details": 3, "what are": 3, "show": 2, "all": 2, "outline": 2,
            # Domain-specific
            "tasks": 3, "action items": 3, "services": 2, "endpoints": 2, "components": 2, "tools": 2, 
            "requirements": 2, "features": 1, "schema": 1, "collections": 1
        }
    }
    
    # Calculate scores for each intent
    scores = {"summarization": 0, "explanation": 0, "list": 0}
    for intent, keywords in intent_keywords.items():
        for keyword, weight in keywords.items():
            if " " in keyword:  # Multi-word phrases
                if keyword in query_lower:
                    scores[intent] += weight
            else:  # Single words
                if keyword in words:
                    scores[intent] += weight
    
    # Determine the top intent
    top_intent = max(scores, key=scores.get)
    max_score = scores[top_intent]
    
    # Robustness: Require a higher threshold and check for ambiguity
    if max_score < 3:  # Low confidence
        return "general"
    elif max_score == scores[max(scores.keys() - {top_intent}, key=scores.get)]:  # Tie between intents
        return "general"  # Fallback if scores are equal
    else:
        return top_intent

# --- Enhanced RAG Query ---
def rag_query(user_id, project, query_text):
    """Generate an optimized RAG prompt with improved intent detection."""
    # Retrieve documents and meeting type
    shared_docs = query_shared_collection(project, query_text, n_results=2)
    user_doc, meeting_type = query_user_collection(user_id, project, query_text, n_results=1)
    
    # Detect intent using the new function
    intent = detect_intent(query_text)
    
    # Meeting Type Instructions
    meeting_instructions = {
        "Technical Meeting": "Focus on technical details, problem-solving steps, and action items.",
        "KT Session": "Explain concepts clearly with examples, focusing on processes or best practices.",
        "Townhall Meeting": "Provide concise insights into announcements, policies, or strategic directions.",
        "Unknown": "Respond based on available data, keeping it relevant and clear."
    }
    base_instruction = meeting_instructions.get(meeting_type, meeting_instructions["Unknown"])

    # Build Instruction Based on Intent
    if intent == "summarization":
        instruction = f"Summarize in 2-3 sentences, {base_instruction.lower()}"
        format_instruction = "Use plain text."
    elif intent == "explanation":
        instruction = f"Explain step-by-step, {base_instruction.lower()}"
        format_instruction = "Use numbered steps if applicable."
    elif intent == "list":
        instruction = f"Provide a detailed response, {base_instruction.lower()}"
        format_instruction = "Use bullet points for key items or steps."
    else:  # General intent
        instruction = f"Answer directly, {base_instruction.lower()}"
        format_instruction = "Use plain text; if the query is unclear, ask for clarification like 'Could you specify what you mean by \"{query_text}\"?'"

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

# # Example usage
# if __name__ == "__main__":
#     queries = [
#         "Summarize the last technical meeting",
#         "How does the deployment process work?",
#         "List the action items from the KT session",
#         "What’s the database schema like?",
#         "Explain the JWT authentication process",
#         "What’s this about?"
#     ]
#     for query in queries:
#         print(f"Query: {query}")
#         print(f"Response: {rag_query('user1', 'ProjectA', query)}\n")