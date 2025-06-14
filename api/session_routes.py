from flask import Blueprint, request, jsonify
from core.session_manager import (
    list_user_sessions,
    rename_session,
    delete_session,
    clear_session_history, 
    get_or_create_session
)
from langchain_core.messages import HumanMessage, AIMessage
from core.rag_engine import rag_query

session_bp = Blueprint('session', __name__)

@session_bp.route('/api/sessions', methods=['GET'])
def get_sessions():
    user_id = request.args.get('user_id')
    project = request.args.get('project')
    print(f"Fetching sessions for user: {user_id}, project: {project}")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
        
    sessions = list_user_sessions(user_id, project)
    return jsonify({"sessions": sessions}), 200

@session_bp.route('/api/sessions/create', methods=['POST'])
def create_session():
    user_id = request.form['user_id']
    project = request.form['project']
    
    result = get_or_create_session(user_id, project, create_new=True)
    print("The result of session creation:", result)
    return jsonify({
        "message": "New session created",
        "session_id": result[1]
    }), 201

@session_bp.route('/api/sessions/rename', methods=['POST'])
def update_session_name():
    """
    Rename a session.
    """
    user_id = request.form['user_id']
    project = request.form['project']
    session_id = request.form['session_id']
    new_name = request.form['name']
    
    success = rename_session(user_id, project, session_id, new_name)
    if success:
        return jsonify({"message": "Session renamed successfully"}), 200
    return jsonify({"error": "Session not found"}), 404

@session_bp.route('/api/sessions/delete', methods=['POST'])
def remove_session():
    """
    Delete a session.
    """
    user_id = request.form['user_id']
    project = request.form['project']
    session_id = request.form['session_id']
    
    success = delete_session(user_id, project, session_id)
    if success:
        return jsonify({"message": "Session deleted successfully"}), 200
    return jsonify({"error": "Session not found"}), 404

@session_bp.route('/api/sessions/clear', methods=['POST'])
def clear_session():
    """
    Clear a session's history.
    """
    user_id = request.form['user_id']
    project = request.form['project']
    session_id = request.form['session_id']
    
    success = clear_session_history(user_id, project, session_id)
    if success:
        return jsonify({"message": "Session history cleared successfully"}), 200
    return jsonify({"error": "Session not found"}), 404

@session_bp.route('/api/sessions/history', methods=['GET'])
def get_session_history():
    """
    Retrieve conversation history for a specific session.
    """
    user_id = request.args.get('user_id')
    project = request.args.get('project')
    session_id = request.args.get('session_id')
    
    if not all([user_id, project, session_id]):
        return jsonify({"error": "User ID, project, and session ID are required"}), 400
    
    try:
        # Get the session memory
        memory, _ = get_or_create_session(user_id, project, session_id)
        
        # Extract messages from memory
        messages = memory.chat_memory.messages
        
        # Convert to the format expected by the UI
        history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                history.append({
                    "role": "assistant",
                    "content": message.content
                })
        
        return jsonify({"history": history}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve session history: {str(e)}"}), 500
