@echo off
echo ========================================
echo Facial Emotion Recognition - WebRTC
echo ========================================
echo.

echo Starting WebRTC version...
echo This version uses modern browser-based video streaming.
echo.

echo Installing required packages...
pip install streamlit-webrtc av

echo.
echo Starting WebRTC application...
python -m streamlit run web_app/app_webrtc.py --server.port 8501

pause
