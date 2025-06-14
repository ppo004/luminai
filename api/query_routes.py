from flask import Blueprint, request, jsonify, Response
import json
from core.rag_engine import rag_query

query_bp = Blueprint('query', __name__)

@query_bp.route('/api/query', methods=['POST'])
def process_query():
    user_id = request.form['user_id']
    project = request.form['project']
    query_text = request.form['query_text']
    
    session_id = request.form.get('session_id')
    stream = request.form.get('stream', 'false').lower() == 'true'
    
    print(f"Query for user: {user_id}, project: {project}, session: {session_id}")
    
    if stream:
        def generate():
            response_generator = rag_query(
                user_id, 
                project, 
                query_text, 
                stream=True, 
                session_id=session_id
            )
            session_identifier = None
            for chunk in response_generator:
                if isinstance(chunk, str) and chunk.startswith('session_id:'):
                    session_identifier = chunk.replace('session_id:', '')
                    yield f"data: {json.dumps({'session_id': session_identifier})}\n\n"
                else:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
        return Response(generate(), mimetype='text/event-stream')
    else:
        result = rag_query(
            user_id, 
            project, 
            query_text, 
            session_id=session_id, 
            create_new=create_new
        )
        return jsonify(result), 200
