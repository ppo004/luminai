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

llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)

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
    print(f"RAG query for project: {project}, user: {user_id}, seesion id: {session_id}")

    memory, active_session_id = get_or_create_session(user_id, project, session_id, create_new)

    vectorstore = get_vectorstore(project)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

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

    chat_history = format_chat_history(memory)

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"])),
            chat_history=lambda x: x["chat_history"]
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    inputs = {
        "input": query_text,
        "instruction_details": instruction_details_str,
        "chat_history": chat_history
    }
    
    increment_message_count(user_id, project, active_session_id)
    
    if stream:
        response_stream = rag_chain.stream(inputs)
        full_response = ""
        stream_generator = (chunk for chunk in response_stream)
        
        for chunk in stream_generator:
            full_response += chunk
            yield chunk
            
        memory.save_context({"input": query_text}, {"answer": full_response})
        
        yield f"session_id:{active_session_id}"

    else:
        response = rag_chain.invoke(inputs)
        memory.save_context({"input": query_text}, {"answer": response})
        return {"response": response, "session_id": active_session_id}
