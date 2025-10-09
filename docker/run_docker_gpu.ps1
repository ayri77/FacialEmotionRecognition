# PowerShell script to run Docker container with GPU support

Write-Host "=" * 100
Write-Host "FACIAL EMOTION RECOGNITION - DOCKER GPU SETUP"
Write-Host "=" * 100

# Check if Docker is running
Write-Host "`nChecking Docker status..."
$dockerStatus = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again."
    exit 1
}
Write-Host "✓ Docker is running" -ForegroundColor Green

# Build Docker image
Write-Host "`nBuilding Docker image with TensorFlow GPU..."
Write-Host "This may take 5-10 minutes on first run..."
Set-Location $PSScriptRoot
docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to build Docker image!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker image built successfully" -ForegroundColor Green

# Start container
Write-Host "`nStarting Docker container..."
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start container!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Container started successfully" -ForegroundColor Green

# Wait a moment for Jupyter to start
Start-Sleep -Seconds 3

# Get container logs to find Jupyter URL
Write-Host "`n" + "=" * 100
Write-Host "JUPYTER NOTEBOOK INFORMATION"
Write-Host "=" * 100
docker-compose logs facial-emotion-gpu | Select-String "http://127.0.0.1:8888"

Write-Host "`n✓ Jupyter Notebook is running at: http://localhost:8888" -ForegroundColor Green
Write-Host "`n" + "=" * 100
Write-Host "USEFUL COMMANDS"
Write-Host "=" * 100
Write-Host "View logs:      docker-compose logs -f facial-emotion-gpu"
Write-Host "Stop container: docker-compose down"
Write-Host "Enter shell:    docker-compose exec facial-emotion-gpu bash"
Write-Host "GPU check:      docker-compose exec facial-emotion-gpu nvidia-smi"
Write-Host "=" * 100
