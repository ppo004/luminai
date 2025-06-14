from flask import Blueprint, request, jsonify
import os
from utils.transcript_processing import process_transcript
import config

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/api/upload', methods=['POST'])
def upload_file():
    user_id = request.form['user_id']
    project = request.form['project']
    meeting_type = request.form['meeting_type']
    file = request.files['transcript']
    
    if file:
        filepath = os.path.join(config.UPLOAD_FOLDER, f"{user_id}_{file.filename}")
        file.save(filepath)
        process_transcript(user_id, project, filepath, meeting_type)
        return jsonify({"message": "Upload successful"}), 200
    
    return jsonify({"error": "No file"}), 400
