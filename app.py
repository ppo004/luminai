"""
Main application entry point.
This file initializes the Flask application and registers all routes.
"""
from flask import Flask
from flask_cors import CORS
import config

# Import route blueprints
from api.query_routes import query_bp
from api.session_routes import session_bp
from api.upload_routes import upload_bp

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: Configured Flask application
    """
    # Initialize Flask app
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Configure app
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
    app.config['DEBUG'] = config.DEBUG
    
    # Register blueprints
    app.register_blueprint(query_bp)
    app.register_blueprint(session_bp)
    app.register_blueprint(upload_bp)
    
    return app

if __name__ == '__main__':
    # Create the application
    app = create_app()
    
    # Run the application
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )