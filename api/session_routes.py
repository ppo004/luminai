"""
Session management endpoints.
"""
from flask import Blueprint, request, jsonify
from core.session_manager import (
    list_user_sessions,
    rename_session,
    delete_session,
    clear_session_history
)
from core.rag_engine import rag_query

# Create blueprint
session_bp = Blueprint('session', __name__)

@session_bp.route('/api/sessions', methods=['GET'])
def get_sessions():
    """
    List all sessions for a user.
    """
    user_id = request.args.get('user_id')
    project = request.args.get('project')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
        
    sessions = list_user_sessions(user_id, project)
    return jsonify({"sessions": sessions}), 200

@session_bp.route('/api/sessions/create', methods=['POST'])
def create_session():
    """
    Create a new session.
    """
    user_id = request.form['user_id']
    project = request.form['project']
    
    # Create a new empty session
    result = rag_query(user_id, project, "", create_new=True)
    return jsonify({
        "message": "New session created",
        "session_id": result["session_id"]
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
