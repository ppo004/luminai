import chromadb
import requests
import json

def query_collection(project, query_text):
    persist_directory = "chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(project)
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )
    return results['documents'][0][0]

def get_llama3_response(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3:8b", "prompt": prompt, "stream": True},  # Enable streaming
        stream=True  # Tell requests to expect a stream
    )
    response.raise_for_status()  # Raise an exception for bad status codes

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode('utf-8'))
                if 'response' in json_data:
                    full_response += json_data['response']
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON chunk: {e}, line: {line.decode('utf-8')}")
                continue  # Skip problematic chunks

    return full_response

def rag_query(project, query_text):
    # Retrieve relevant document from the collection
    relevant_doc = query_collection(project, query_text)
    # Combine document and query into a prompt
    prompt = f"Context: {relevant_doc}\n\nQuery: {query_text}\n\nResponse:"
    # Get response from Llama 3
    response = get_llama3_response(prompt)
    return response

# Test the RAG logic
if __name__ == "__main__":
    project = "ProjectA"
    query = "What is the main topic?"
    response = rag_query(project, query)
    print(response)
