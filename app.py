from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import rag_query
from process_transcript import process_transcript
import os

app = Flask(__name__)
CORS(app)  # Allow Angular to connect
app.config['UPLOAD_FOLDER'] = 'transcripts'

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
    res = rag_query(user_id, project, query_text)
    print(res)
    response = {"response": f"Processed query: {res}"}
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)