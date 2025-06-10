"""
Core business logic package.
"""
from core.rag_engine import rag_query
from core.intent_detection import detect_intent, get_instruction_and_format
from core.session_manager import (
    get_or_create_session,
    list_user_sessions,
    rename_session,
    delete_session,
    clear_session_history,
    format_chat_history,
    increment_message_count
)

__all__ = [
    'rag_query',
    'detect_intent',
    'get_instruction_and_format',
    'get_or_create_session',
    'list_user_sessions',
    'rename_session',
    'delete_session',
    'clear_session_history',
    'format_chat_history',
    'increment_message_count'
]
