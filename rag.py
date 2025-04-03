import chromadb
import requests
import json

def query_collections(user_id, project, query_text):
    # Set up persistent ChromaDB client
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Query shared collection
    shared_collection = client.get_collection(f"{project}_shared")
    shared_results = shared_collection.query(query_texts=[query_text], n_results=1)
    shared_doc = shared_results['documents'][0][0] if shared_results['documents'] else ""
    
    # Query user-specific collection (if it exists)
    user_collection_name = f"{project}_{user_id}"
    user_doc = ""
    try:
        user_collection = client.get_collection(user_collection_name)
        user_results = user_collection.query(query_texts=[query_text], n_results=1)
        user_doc = user_results['documents'][0][0] if user_results['documents'] else ""
    except ValueError:
        # Collection doesn’t exist for this user yet
        pass
    
    return f"Shared: {shared_doc}\nUser: {user_doc}"

def get_llama3_response(prompt):
    # Call LLaMA3 API (assumes it’s running locally)
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

def rag_query(user_id, project, query_text):
    # Get context from shared and user data
    context = query_collections(user_id, project, query_text)
    prompt = f"Context: {context}\n\nQuery: {query_text}\n\nResponse:"
    return get_llama3_response(prompt)

# Example usage
if __name__ == "__main__":
    user_id = "user1"
    project = "ProjectA"
    query = "Who fought in Battle of Purandar?"
    response = rag_query(user_id, project, query)
    print(response)
