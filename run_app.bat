@echo off
echo Starting Facial Emotion Recognition App...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the application
python -m streamlit run streamlit_app.py

pause
