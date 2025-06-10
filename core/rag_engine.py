"""
RAG (Retrieval Augmented Generation) engine.
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config
from core.intent_detection import detect_intent, get_instruction_and_format
from core.session_manager import (
    get_or_create_session,
    format_chat_history,
    increment_message_count,
    clear_session_history
)
from utils.embedding_utils import get_vectorstore, format_docs

# Initialize LLM
llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)

# Define prompt template
prompt_template_str = """You are an assistant for project onboarding, documentation, PBIs, HR, and internal tools.
{instruction_details}
Keep responses professional, concise, and relevant. Define technical terms if needed.

**Conversation History**:
{chat_history}

**Shared Project Data for current query (Context)**:
{context}

**Response**:
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template_str),
    ("human", "{input}")
])

def rag_query(user_id, project, query_text, stream=False, session_id=None, create_new=False):
    """
    Perform a RAG query with conversation memory.
    
    Args:
        user_id (str): User identifier
        project (str): Project identifier
        query_text (str): User query
        stream (bool): Whether to stream the response
        session_id (str, optional): Specific session ID to use
        create_new (bool): Whether to create a new session
        
    Returns:
        dict or generator: Response data or stream
    """
    print(f"RAG query for project: {project}, user: {user_id}")
    
    # Check if query contains session management commands
    if query_text.lower().startswith("/new_session"):
        create_new = True
        # Extract the actual query if provided after command
        parts = query_text.split(" ", 1)
        if len(parts) > 1:
            query_text = parts[1]
        else:
            return "Started a new conversation session. What would you like to know?"
    
    elif query_text.lower().startswith("/clear_history"):
        clear_session_history(user_id, project, session_id)
        parts = query_text.split(" ", 1)
        if len(parts) > 1:
            query_text = parts[1]
        else:
            return "Conversation history cleared. What would you like to know?"
    
    # Get or create session based on parameters
    memory, active_session_id = get_or_create_session(user_id, project, session_id, create_new)
    
    # Get vector store and retriever
    vectorstore = get_vectorstore(project)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Detect intent and get formatting instructions
    intent = detect_intent(query_text)
    print(f"Detected intent: {intent}")

    instruction, format_instruction = get_instruction_and_format(intent)
    priority_instruction = "Use the shared project data to answer the query."
    
    instruction_details_str = (
        f"**Instructions for the current query**:\n"
        f"- {priority_instruction}\n"
        f"- {instruction}\n"
        f"- {format_instruction}\n"
        f"- Consider the conversation history for context"
    )

    # Format chat history for inclusion in the prompt
    chat_history = format_chat_history(memory)

    # Build the RAG chain
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"])),
            chat_history=lambda x: x["chat_history"]
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Prepare inputs
    inputs = {
        "input": query_text,
        "instruction_details": instruction_details_str,
        "chat_history": chat_history
    }
    
    # Increment message count
    increment_message_count(user_id, project, active_session_id)
    
    # Handle streaming response
    if stream:
        response_stream = rag_chain.stream(inputs)
        full_response = ""
        stream_generator = (chunk for chunk in response_stream)
        
        for chunk in stream_generator:
            full_response += chunk
            yield chunk
            
        # Add to memory after streaming is complete
        memory.save_context({"input": query_text}, {"answer": full_response})
        
        # Return the session ID as the last message
        yield f"session_id:{active_session_id}"
    
    # Handle non-streaming response
    else:
        response = rag_chain.invoke(inputs)
        
        # Add the exchange to memory
        memory.save_context({"input": query_text}, {"answer": response})
        
        return {"response": response, "session_id": active_session_id}
