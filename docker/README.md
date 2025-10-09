# Docker GPU Setup for Facial Emotion Recognition

This directory contains Docker configuration for running the project with GPU acceleration.

## 🐳 Quick Start

### Prerequisites
- Docker Desktop with WSL2 backend
- NVIDIA GPU with recent drivers
- NVIDIA Container Toolkit

### Launch

**Windows (PowerShell):**
```powershell
cd docker
.\run_docker_gpu.ps1
```

**Manual:**
```bash
cd docker
docker-compose up -d
```

Access Jupyter at: http://localhost:8888
Access TensorBoard at: http://localhost:6006

## 📁 Files

- `Dockerfile` - Docker image definition with TensorFlow GPU
- `docker-compose.yml` - Container orchestration with GPU support
- `run_docker_gpu.ps1` - Windows PowerShell launch script
- `.dockerignore` - Files to exclude from Docker context

## 🔧 Useful Commands

```bash
# View logs
docker-compose logs -f facial-emotion-gpu

# Check GPU
docker exec facial-emotion-recognition-gpu nvidia-smi

# Enter container shell
docker exec -it facial-emotion-recognition-gpu bash

# Stop container
docker-compose down

# Rebuild image
docker-compose build --no-cache
```

## 📊 TensorBoard Monitoring

**Start TensorBoard in Docker:**
```bash
docker exec -d facial-emotion-recognition-gpu tensorboard --logdir=/workspace/runs/tensorboard --host=0.0.0.0 --port=6006
```

**Access TensorBoard:**
- Open browser: http://localhost:6006
- View real-time training metrics
- Compare all models side-by-side
- Analyze training curves, histograms, and graphs

**What you can monitor:**
- Training/validation accuracy and loss curves
- Learning rate schedules
- Model architecture graphs
- Weight and gradient distributions
- Per-epoch metrics for all models

## ⚙️ Configuration

The Docker setup includes:
- TensorFlow 2.15.0 with GPU support
- CUDA 11.8 compatibility
- All project dependencies
- Jupyter Notebook server
- Automatic GPU memory growth
- Volume mounts for data persistence

## 🎯 Expected Performance

With NVIDIA RTX 4080:
- Baseline CNN: ~3-5 minutes
- Deep Regularized: ~5-10 minutes
- VGG16: ~15-25 minutes
- ResNet50V2: ~15-25 minutes
- EfficientNet: ~10-20 minutes
- Complex CNN: ~20-40 minutes

**Total training time: ~1.5-2.5 hours** (vs 12-24 hours on CPU)

## 🐛 Troubleshooting

**GPU not detected:**
- Ensure Docker Desktop is using WSL2 backend
- Check NVIDIA Container Toolkit is installed
- Verify GPU is visible: `nvidia-smi`

**Container won't start:**
- Check Docker Desktop is running
- Verify ports are not in use: `netstat -an | findstr :8888`
- Check logs: `docker-compose logs`

**Permission errors:**
- Run PowerShell as Administrator
- Check Docker has access to project directory
