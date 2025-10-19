# Docker Testing Environment for Facial Emotion Recognition App

This directory contains a Docker environment that mimics Streamlit Cloud for testing the Facial Emotion Recognition application.

## Environment Details

- **Base Image**: `python:3.10-slim` (same as Streamlit Cloud)
- **Python Version**: 3.10 (for compatibility with all libraries)
- **Dependencies**: All packages from `requirements.txt`
- **System Libraries**: Same as Streamlit Cloud environment

## Quick Start

### Using Docker Compose (Recommended)

**Windows (PowerShell):**
```powershell
.\run_test.ps1
```

**Linux/Mac:**
```bash
chmod +x run_test.sh
./run_test.sh
```

### Manual Commands

1. **Build the image:**
   ```bash
   docker-compose build
   ```

2. **Start the application:**
   ```bash
   docker-compose up
   ```

3. **Access the app:**
   Open http://localhost:8501 in your browser

4. **Stop the application:**
   ```bash
   docker-compose down
   ```

## What to Test

1. **Model Download**: Click "🧪 Check model file" button to test Hugging Face model download
2. **WebRTC Mode**: Test real-time emotion detection
3. **Photo Mode**: Test static image emotion recognition
4. **Dependencies**: Verify all libraries work correctly
5. **Performance**: Check if the app runs smoothly in containerized environment

## Debugging

- **Logs**: Check Docker logs with `docker-compose logs -f`
- **Model Cache**: Models are cached in `./models` directory
- **Health Check**: Container health is monitored automatically

## Troubleshooting

- If model download fails, check internet connectivity
- If WebRTC doesn't work, it's expected in Docker (camera access limitations)
- Check logs for detailed error messages

## Environment Variables

The container uses the same environment variables as Streamlit Cloud:
- `PYTHONUNBUFFERED=1`
- `TF_CPP_MIN_LOG_LEVEL=3`
- `TF_ENABLE_ONEDNN_OPTS=0`
