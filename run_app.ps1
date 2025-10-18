# Facial Emotion Recognition - PowerShell Launcher
Write-Host "🚀 Starting Facial Emotion Recognition App..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Run the application
Write-Host "🌐 Starting Streamlit application..." -ForegroundColor Yellow
Write-Host "⏹️  Press Ctrl+C to stop the application" -ForegroundColor Cyan
Write-Host ""

python -m streamlit run streamlit_app.py
