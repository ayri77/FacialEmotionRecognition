"""
Facial Emotion Recognition - Streamlit Cloud App
Main entry point for Streamlit Cloud deployment
"""

import sys

# Add web_app directory to path
sys.path.append("web_app")

# Import the main app
from main_app import main

if __name__ == "__main__":
    main()
