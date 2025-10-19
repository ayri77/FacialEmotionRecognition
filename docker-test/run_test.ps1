# PowerShell script to test Facial Emotion Recognition App in Docker
Write-Host "🐳 Building Docker image for Facial Emotion Recognition App testing..." -ForegroundColor Green
Write-Host "This will test the app in an environment similar to Streamlit Cloud" -ForegroundColor Yellow

# Build the image
Write-Host "Building Docker image..." -ForegroundColor Cyan
docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Starting the application..." -ForegroundColor Green
Write-Host "The app will be available at: http://localhost:8501" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow

# Start the container
docker-compose up

Write-Host "🛑 Stopping containers..." -ForegroundColor Green
docker-compose down
