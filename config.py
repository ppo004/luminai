"""
Configuration settings for the LuminAI application.
"""
import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODELS_DIRECTORY = os.path.join(BASE_DIR, "models")

# LLM settings
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# API settings
DEBUG = True
HOST = "0.0.0.0"
PORT = 5001
