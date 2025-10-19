"""
Facial Emotion Recognition - Streamlit Cloud App
Main entry point for Streamlit Cloud deployment
"""

import streamlit as st

# Page configuration - MUST be first command
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys  # noqa: E402

# Add web_app directory to path
sys.path.append("web_app")

# Import the main app
from main_app import main  # noqa: E402

if __name__ == "__main__":
    main()
