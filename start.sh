#!/bin/bash
# Start script for LuminAI

echo "Starting LuminAI..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the Flask server in the background
echo "Starting Flask server on port 5001..."
python3 run.py &
SERVER_PID=$!

# Wait a moment for the server to start
sleep 2

# Start the Streamlit UI
echo "Starting Streamlit UI on port 8501..."
streamlit run app_ui.py

# Cleanup when Streamlit is closed
kill $SERVER_PID
echo "LuminAI stopped."
