#!/usr/bin/env python3
"""
Run script for LuminAI application.
"""
from app import create_app
import config

if __name__ == "__main__":
    # Create the application
    app = create_app()
    
    # Run the application
    print(f"Starting LuminAI server on http://{config.HOST}:{config.PORT}")
    print("Press CTRL+C to quit")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
