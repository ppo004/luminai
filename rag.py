import chromadb
import requests
import json
import re
from utils import embedding_model

# --- Query Functions ---

def query_shared_collection(project, query_text=None, query_embeddings=None, n_results=2):
    """
    Retrieve documents from the shared project collection using either query text or embeddings.
    
    Args:
        project (str): The project identifier.
        query_text (str, optional): The query text to embed and search.
        query_embeddings (list, optional): List of embeddings to query directly.
        n_results (int): Number of results to return (per embedding if query_embeddings is used).
    
    Returns:
        str: Formatted string of retrieved documents or an error message.
    """
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(f"{project}_shared")
        if query_text:
            query_embedding = embedding_model.encode(query_text).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["documents"])
            if results['documents'] and results['documents'][0]:
                return "\n".join([f"- {doc}" for doc in results['documents'][0]])
        elif query_embeddings:
            results = collection.query(query_embeddings=query_embeddings, n_results=1, include=["documents"])
            documents = [doc for sublist in results['documents'] for doc in sublist]
            return "\n".join([f"- {doc}" for doc in documents])
        return "No shared project data available."
    except ValueError:
        return "Shared project data not initialized."

def query_user_collection(user_id, project, query_text, n_results=3, meeting_type_filter=None, retrieve_all=False):
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = f"{project}_{user_id}"
    try:
        collection = client.get_collection(collection_name)
        where_clause = {"meeting_type": meeting_type_filter} if meeting_type_filter else None
        if retrieve_all:
            # Retrieve all documents matching the meeting_type filter
            results = collection.get(where=where_clause, include=["documents", "metadatas"])
            docs = results['documents']
            meeting_types = [meta.get("meeting_type", "Unknown") for meta in results['metadatas']]
            embeddings = []  # Embeddings not needed for summarization
        else:
            # Existing similarity search logic
            query_embedding = embedding_model.encode(query_text).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "embeddings"]
            )
            if results['documents'] and results['documents'][0]:
                docs = results['documents'][0]
                embeddings = results['embeddings'][0]
                meeting_types = [meta.get("meeting_type", "Unknown") for meta in results['metadatas'][0]]
            else:
                docs = []
                embeddings = []
                meeting_types = []
        
        if docs:
            user_doc = "\n".join([f"- {doc}" for doc in docs])
            meeting_type = meeting_types[0] if meeting_types else "Unknown"
            return user_doc, meeting_type, embeddings
        return "No user-specific data available.", "Unknown", []
    except Exception:
        return "No user-specific data available.", "Unknown", []
# --- Llama 3 Response ---

def get_llama3_response(prompt, stream=False):
    """
    Call Llama 3 API with option for streaming response.
    
    Args:
        prompt (str): The prompt to send to the API.
        stream (bool): Whether to stream the response.
    
    Returns:
        str or generator: Full response string or generator for streaming.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3:8b", "prompt": prompt, "stream": True},
            stream=True
        )
        response.raise_for_status()
        
        if not stream:
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
        else:
            def generate():
                for line in response.iter_lines():
                    if line:
                        try:
                            json_data = json.loads(line.decode('utf-8'))
                            if 'response' in json_data:
                                yield json_data['response']
                        except json.JSONDecodeError:
                            continue
            return generate()
    except requests.RequestException as e:
        if stream:
            def error_generator():
                yield f"Error contacting Llama 3 API: {str(e)}"
            return error_generator()
        else:
            return f"Error contacting Llama 3 API: {str(e)}"

# --- Robust Intent Detection with Domain-Specific Keywords ---

def detect_intent(query_text):
    """
    Detect query intent using weighted keywords with domain-specific terms.
    
    Args:
        query_text (str): The user's query text.
    
    Returns:
        str: Detected intent ("summarization", "explanation", "list", or "general").
    """
    query_lower = " ".join(query_text.lower().split())
    words = query_lower.split()
    
    intent_keywords = {
        "summarization": {
            "summarize": 4, "summary": 4, "overview": 3, "brief": 3, "give me a": 2, "short": 2, "recap": 2,
            "status": 2, "progress": 2, "update": 2, "meeting": 1, "project": 1
        },
        "explanation": {
            "explain": 4, "how": 4, "why": 4, "what is": 3, "break down": 3, "describe": 2, "tell me": 2, "clarify": 2,
            "works": 3, "process": 3, "deploy": 2, "authentication": 2, "jwt": 2, "database": 2, "api": 2, 
            "microservices": 2, "integration": 1, "setup": 1, "function": 1
        },
        "list": {
            "list": 4, "steps": 4, "items": 3, "details": 3, "what are": 3, "show": 2, "all": 2, "outline": 2,
            "tasks": 3, "action items": 3, "services": 2, "endpoints": 2, "components": 2, "tools": 2, 
            "requirements": 2, "features": 1, "schema": 1, "collections": 1
        }
    }
    
    scores = {"summarization": 0, "explanation": 0, "list": 0}
    for intent, keywords in intent_keywords.items():
        for keyword, weight in keywords.items():
            if " " in keyword:
                if keyword in query_lower:
                    scores[intent] += weight
            else:
                if keyword in words:
                    scores[intent] += weight
    print("The intent scores", scores)
    top_intent = max(scores, key=scores.get)
    max_score = scores[top_intent]
    
    if max_score < 3:
        return "general"
    elif max_score == scores[max(scores.keys() - {top_intent}, key=scores.get)]:
        return "general"
    else:
        return top_intent

# --- Enhanced RAG Query ---

def rag_query(user_id, project, query_text, stream=False):
    """
    Generate an optimized RAG prompt with improved intent detection and transcript prioritization.
    
    Args:
        user_id (str): The user identifier.
        project (str): The project identifier.
        query_text (str): The user's query text.
        stream (bool): Whether to stream the response.
    
    Returns:
        str or generator: Response from Llama 3 API.
    """
    intent = detect_intent(query_text)
    is_transcript_specific = intent == "summarization" and any(kw in query_text.lower() for kw in ["meeting", "transcript", "session", "discussion", "video"])
    print("Is it video specific:",is_transcript_specific)
    # Fetch user data with embeddings
    meeting_type_filter = None
    if "technical" in query_text.lower():
        meeting_type_filter = "Technical Meeting"
    elif "kt" in query_text.lower():
        meeting_type_filter = "KT Session"
    
    # Fetch shared data
    if is_transcript_specific:
        shared_docs = None
        user_doc, meeting_type, _ = query_user_collection(
            user_id, project, query_text, 
            meeting_type_filter=meeting_type_filter, 
            retrieve_all=True
        )
    else:
        shared_docs = query_shared_collection(project, query_text=query_text, n_results=3)
        user_doc, meeting_type, user_embeddings = query_user_collection(
            user_id, project, query_text, 
            n_results=5, 
            meeting_type_filter=meeting_type_filter
        )
    print("Shared docs", shared_docs)
    print("User Docs", user_doc)
    # Build instructions
    meeting_instructions = {
        "Technical Meeting": "Focus on technical details, problem-solving steps, and action items.",
        "KT Session": "Explain concepts clearly with examples, focusing on processes or best practices.",
        "Townhall Meeting": "Provide concise insights into announcements, policies, or strategic directions.",
        "Unknown": "Respond based on available data, keeping it relevant and clear."
    }
    base_instruction = meeting_instructions.get(meeting_type, meeting_instructions["Unknown"])
    
    if intent == "summarization":
        instruction = f"Summarize in 6-7 sentences, {base_instruction.lower()}"
        format_instruction = "Use plain text."
    elif intent == "explanation":
        instruction = f"Explain step-by-step, {base_instruction.lower()}"
        format_instruction = "Use numbered steps if applicable."
    elif intent == "list":
        instruction = f"Provide a detailed response, {base_instruction.lower()}"
        format_instruction = "Use bullet points for key items or steps."
    else:
        instruction = f"Answer directly, {base_instruction.lower()}"
        format_instruction = "Use plain text; if the query is unclear, ask for clarification."
    
    # Prioritization instruction
    if is_transcript_specific:
        priority_instruction = "Prioritize the data below, which is from the transcript"
        prompt = f"""You are an IT assistant for technical meetings, KT sessions, and townhalls.  
    **Instructions**:  
    - {priority_instruction}  
    - {instruction}  
    - {format_instruction}  
    - Keep responses professional, concise, and relevant. Define technical terms if needed.  
    **User Data ({meeting_type})**:  
    {user_doc}  
    **Query**:  
    {query_text}  
    **Response**:  
    """
    else:
        priority_instruction = "Use both the user data and shared data to answer the query."
    
        prompt = f"""You are an IT assistant for technical meetings, KT sessions, and townhalls.  
    **Instructions**:  
    - {priority_instruction}  
    - {instruction}  
    - {format_instruction}  
    - Keep responses professional, concise, and relevant. Define technical terms if needed.  
    **User Data ({meeting_type})**:  
    {user_doc}  
    **Shared Project Data**:  
    {shared_docs}  
    **Query**:  
    {query_text}  
    **Response**:  
    """
    print("The prompt", prompt)
    return get_llama3_response(prompt, stream=stream)