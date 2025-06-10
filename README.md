# LuminAI

A RAG-based AI assistant for project documentation and onboarding.

## Project Structure

```
LuminAI/
├── api/                      # API endpoints
│   ├── __init__.py           # Makes the directory a package
│   ├── query_routes.py       # RAG query endpoints
│   ├── session_routes.py     # Session management endpoints
│   └── upload_routes.py      # File upload endpoints
├── core/                     # Core business logic
│   ├── __init__.py
│   ├── intent_detection.py   # Intent detection logic
│   ├── rag_engine.py         # RAG functionality
│   └── session_manager.py    # Session management
├── data/                     # Data files
├── models/                   # Model files (embeddings, etc.)
├── seeders/                  # Data seeders
│   ├── __init__.py
│   ├── chromadb_seeder.py    # ChromaDB initialization
│   └── ad_data_seeder.py     # AD data seeding
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── embedding_utils.py    # Embedding-related utilities
│   └── text_processing.py    # Text processing utilities
├── app.py                    # Main application file
├── config.py                 # Configuration settings
├── run.py                    # Run script
└── requirements.txt          # Dependencies
```

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Configure settings in `config.py`

3. Initialize the ChromaDB:
```
python -m seeders.chromadb_seeder
```

4. Start the application:
```
# Option 1: Using the start script (recommended)
./start.sh

# Option 2: Starting components separately
./run.py  # Start the API server
streamlit run app_ui.py  # Start the UI in a separate terminal
```

## API Endpoints

### Querying

- `POST /api/query` - Submit a RAG query

### Session Management

- `GET /api/sessions` - List sessions
- `POST /api/sessions/create` - Create a new session
- `POST /api/sessions/rename` - Rename a session
- `POST /api/sessions/delete` - Delete a session
- `POST /api/sessions/clear` - Clear session history

### File Upload

- `POST /api/upload` - Upload and process a transcript file

## Development

### Adding New Endpoints

1. Create a new route file in the `api/` directory
2. Register the blueprint in `app.py`

### Working with Sessions

Sessions are managed through the `core/session_manager.py` module. Each session contains:
- Conversation memory
- Creation timestamp
- Last accessed timestamp
- Message count
- Custom name (optional)

## Configuration

See `config.py` for available configuration options.

## Requirements

- Python 3.8+
- Ollama (with Llama3 and Nomic Embed models)
- Flask
- LangChain
- ChromaDB
