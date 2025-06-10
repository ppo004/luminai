"""
Session management module for conversation memory.
"""
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import time
from datetime import datetime

# Dictionary to store user sessions
user_sessions = {}

def generate_session_id():
    """Generate a unique session ID based on timestamp and random string"""
    timestamp = int(time.time())
    random_suffix = uuid.uuid4().hex[:8]  # 8 characters from UUID
    return f"{timestamp}_{random_suffix}"

def get_or_create_session(user_id, project, session_id=None, create_new=False):
    """
    Get or create a conversation session.
    
    Args:
        user_id (str): User identifier
        project (str): Project identifier
        session_id (str, optional): Specific session ID to retrieve
        create_new (bool): Whether to create a new session regardless
        
    Returns:
        tuple: (ConversationBufferMemory, session_id)
    """
    # Initialize user's session container if not exists
    if user_id not in user_sessions:
        user_sessions[user_id] = {}
    
    if project not in user_sessions[user_id]:
        user_sessions[user_id][project] = {}
    
    # Create new session if requested or none exists
    if create_new or not user_sessions[user_id][project]:
        new_session_id = session_id or generate_session_id()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        user_sessions[user_id][project][new_session_id] = {
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            "created_at": timestamp,
            "last_accessed": timestamp,
            "message_count": 0,
            "name": f"Session {timestamp}"  # Default name based on timestamp
        }
        print(f"Created new session {new_session_id} for user {user_id}, project {project}")
        return user_sessions[user_id][project][new_session_id]["memory"], new_session_id
    
    # Retrieve specific session if requested
    if session_id and session_id in user_sessions[user_id][project]:
        session = user_sessions[user_id][project][session_id]
        session["last_accessed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return session["memory"], session_id
    
    # Get the most recently accessed session if no specific one requested
    most_recent = sorted(
        user_sessions[user_id][project].items(),
        key=lambda x: x[1]["last_accessed"],
        reverse=True
    )[0][0]
    
    user_sessions[user_id][project][most_recent]["last_accessed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return user_sessions[user_id][project][most_recent]["memory"], most_recent

def list_user_sessions(user_id, project=None):
    """
    List all sessions for a user, optionally filtered by project.
    
    Args:
        user_id (str): User identifier
        project (str, optional): Project to filter by
        
    Returns:
        dict: Dictionary of sessions with metadata
    """
    if user_id not in user_sessions:
        return {}
    
    if project:
        if project not in user_sessions[user_id]:
            return {}
        
        # Return sessions for specific project with metadata
        result = {}
        for session_id, session_data in user_sessions[user_id][project].items():
            result[session_id] = {
                "created_at": session_data["created_at"],
                "last_accessed": session_data["last_accessed"],
                "message_count": session_data["message_count"],
                "name": session_data.get("name", f"Session {session_data['created_at']}")
            }
        return result
    
    # Return sessions across all projects
    result = {}
    for proj, sessions in user_sessions[user_id].items():
        result[proj] = {}
        for session_id, session_data in sessions.items():
            result[proj][session_id] = {
                "created_at": session_data["created_at"],
                "last_accessed": session_data["last_accessed"],
                "message_count": session_data["message_count"],
                "name": session_data.get("name", f"Session {session_data['created_at']}")
            }
    return result

def rename_session(user_id, project, session_id, new_name):
    """
    Rename a specific session.
    
    Args:
        user_id (str): User identifier
        project (str): Project identifier
        session_id (str): Session identifier
        new_name (str): New name for the session
        
    Returns:
        bool: Success status
    """
    if (user_id in user_sessions and 
        project in user_sessions[user_id] and 
        session_id in user_sessions[user_id][project]):
        
        user_sessions[user_id][project][session_id]["name"] = new_name
        return True
    return False

def delete_session(user_id, project, session_id):
    """
    Delete a specific session.
    
    Args:
        user_id (str): User identifier
        project (str): Project identifier
        session_id (str): Session identifier
        
    Returns:
        bool: Success status
    """
    if (user_id in user_sessions and 
        project in user_sessions[user_id] and 
        session_id in user_sessions[user_id][project]):
        
        del user_sessions[user_id][project][session_id]
        return True
    return False

def clear_session_history(user_id, project, session_id):
    """
    Clear the conversation history of a specific session.
    
    Args:
        user_id (str): User identifier
        project (str): Project identifier
        session_id (str): Session identifier
        
    Returns:
        bool: Success status
    """
    if (user_id in user_sessions and 
        project in user_sessions[user_id] and 
        session_id in user_sessions[user_id][project]):
        
        # Reset the memory while preserving metadata
        session_data = user_sessions[user_id][project][session_id]
        session_data["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        session_data["message_count"] = 0
        return True
    return False

def increment_message_count(user_id, project, session_id):
    """
    Increment the message count for a session.
    
    Args:
        user_id (str): User identifier
        project (str): Project identifier
        session_id (str): Session identifier
        
    Returns:
        bool: Success status
    """
    if (user_id in user_sessions and 
        project in user_sessions[user_id] and 
        session_id in user_sessions[user_id][project]):
        
        user_sessions[user_id][project][session_id]["message_count"] += 1
        user_sessions[user_id][project][session_id]["last_accessed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True
    return False

def format_chat_history(memory):
    """
    Format the chat history from memory into a readable string.
    
    Args:
        memory: ConversationBufferMemory object
        
    Returns:
        str: Formatted chat history
    """
    messages = memory.chat_memory.messages
    formatted_history = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted_history += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"AI: {message.content}\n"
    return formatted_history if formatted_history else "No previous conversation."
