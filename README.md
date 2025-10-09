# Facial Emotion Recognition

Deep Learning project for facial emotion recognition using CNN and Transfer Learning approaches.

## 📊 Project Overview

This project implements and compares multiple deep learning architectures for recognizing facial emotions from grayscale images. The dataset contains four emotion classes: happy, sad, neutral, and surprise.

## 🎯 Objectives

- Build custom CNN architectures optimized for emotion recognition
- Compare Transfer Learning approaches (VGG16, ResNet50V2, EfficientNetV2B2)
- Analyze performance trade-offs between model complexity and accuracy
- Handle class imbalance in emotion datasets
- Evaluate models using comprehensive metrics (accuracy, F1-scores, confusion matrices)

## 📁 Project Structure

```
FacialEmotionRecognition/
├── data/                          # Dataset directory
│   ├── docs/                      # Problem statement and documentation
│   ├── train/                     # Training images
│   ├── validation/                # Validation images
│   └── test/                      # Test images
│
├── notebooks/                     # Jupyter notebooks
│   ├── Facial_Emotion_Recognition_Complete.ipynb    # Main analysis notebook
│   └── Facial_Emotion_Recognition_Template.ipynb    # Template notebook
│
├── models/                        # Saved model checkpoints
├── runs/                          # Training logs and artifacts
├── reports/                       # Model comparison reports
│
├── docker/                        # Docker configuration
│   ├── Dockerfile                 # Docker image definition
│   ├── docker-compose.yml         # Docker Compose configuration
│   ├── run_docker_gpu.ps1         # Windows launch script
│   └── .dockerignore              # Docker ignore file
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## 🚀 Quick Start

### Option 1: Local Environment (CPU)

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Option 2: Docker with GPU (Recommended)

```bash
# Navigate to docker directory
cd docker

# Build and run Docker container
docker-compose up -d

# Access Jupyter at http://localhost:8888
```

**Useful Docker commands:**
```bash
# View logs
docker logs facial-emotion-recognition-gpu

# Check GPU
docker exec facial-emotion-recognition-gpu nvidia-smi

# Start TensorBoard monitoring
docker exec -d facial-emotion-recognition-gpu tensorboard --logdir=/workspace/runs/tensorboard --host=0.0.0.0 --port=6006

# Stop container
docker-compose down
```

**Access Services:**
- Jupyter Notebook: http://localhost:8888
- TensorBoard: http://localhost:6006

## 🧠 Models Implemented

### Custom CNN Architectures
1. **Baseline CNN** - 3 blocks, 288K parameters
2. **Deep Regularized CNN** - 4 blocks with strong regularization, 1.96M parameters
3. **Complex Custom CNN** - 5 blocks, heavy architecture, ~8-12M parameters

### Transfer Learning Models
4. **VGG16** - Pre-trained on ImageNet
5. **ResNet50V2** - Residual connections for deeper networks
6. **EfficientNetV2B2** - Compound scaling for efficiency

## 📈 Key Features

- **Class Imbalance Handling**: Computed class weights for balanced training
- **Data Augmentation**: Horizontal flip, brightness, contrast adjustments
- **Comprehensive Evaluation**: Accuracy, Macro F1, Weighted F1, Confusion matrices
- **Experiment Tracking**: Automated logging of hyperparameters, metrics, and artifacts
- **TensorBoard Monitoring**: Real-time visualization of training progress
- **GPU Acceleration**: Docker setup with NVIDIA GPU support

## 📊 Metrics & Evaluation

Each model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Macro F1-Score**: Unweighted mean F1 across classes
- **Weighted F1-Score**: Weighted by class frequency
- **Confusion Matrix**: Detailed per-class performance
- **Training Time**: Efficiency measurement

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Metrics and evaluation
- **Docker**: GPU-enabled containerization
- **Jupyter**: Interactive development

## 📝 Dataset

The dataset consists of 48×48 grayscale facial images across four emotion categories:
- **Training**: ~15,000 images
- **Validation**: ~5,000 images
- **Test**: 128 images (balanced)

**Classes:**
- Happy
- Sad
- Neutral
- Surprise

## 🏆 Results

Detailed results and model comparisons are available in the `reports/` directory after training.

## 📖 Usage

1. **Exploratory Data Analysis**: Analyze class distribution and visualize samples
2. **Model Training**: Train multiple architectures with different configurations
3. **Evaluation**: Compare models using comprehensive metrics
4. **Model Selection**: Choose best model based on test performance

## 🔧 Requirements

- **Python**: 3.11+ (for GPU) or 3.13+ (for CPU)
- **TensorFlow**: 2.15+ with GPU support (Docker) or 2.20+ (CPU)
- **CUDA**: 11.8+ (for Docker GPU)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended

## 📚 References

- TensorFlow Documentation
- Keras Applications (Pre-trained Models)
- scikit-learn Metrics

## 👥 Author

MIT Capstone Project - Facial Emotion Recognition

## 📄 License

Educational project for MIT course.

---

**Note**: For GPU training, Docker with NVIDIA Container Toolkit is recommended on Windows.
