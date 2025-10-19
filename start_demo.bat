@echo off
echo ========================================
echo Facial Emotion Recognition - Demo Server
echo ========================================
echo.

echo Starting minimal version (no video processing)...
echo This version avoids all recursion issues.
echo.

python -m streamlit run web_app/app_minimal.py --server.port 8501

pause
