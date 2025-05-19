from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from rag import rag_query
from process_transcript import process_transcript
import os
import json

app = Flask(__name__)
CORS(app)  # Allow Angular to connect
app.config['UPLOAD_FOLDER'] = 'data'

@app.route('/api/upload', methods=['POST'])
def upload():
    user_id = request.form['user_id']
    project = request.form['project']
    meeting_type = request.form['meeting_type']
    file = request.files['transcript']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_{file.filename}")
        file.save(filepath)
        process_transcript(user_id, project, filepath, meeting_type)
        return jsonify({"message": "Upload successful"}), 200
    return jsonify({"error": "No file"}), 400

@app.route('/api/query', methods=['POST'])
def query():
    user_id = request.form['user_id']
    project = request.form['project']
    query_text = request.form['query_text']
    print(user_id, project, query_text)

    # Check if streaming is requested
    stream = request.form.get('stream', 'false').lower() == 'true'
    
    if stream:
        def generate():
            response_generator = rag_query(user_id, project, query_text, stream=True)
            for chunk in response_generator:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        return Response(generate(), mimetype='text/event-stream')
    else:
        res = rag_query(user_id, project, query_text)
        return jsonify({"response": res}), 200
    
if __name__ == '__main__':
    app.run(debug=True)