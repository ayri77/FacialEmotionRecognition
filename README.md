# Facial Emotion Recognition System

A comprehensive deep learning system for real-time facial emotion recognition using Convolutional Neural Networks (CNNs) and Transfer Learning.

## 🎯 Project Overview

This project implements a state-of-the-art facial emotion recognition system that can detect and classify emotions in real-time using webcam input. The system supports 4 emotion classes: **Happy**, **Neutral**, **Sad**, and **Surprise**.

## 🚀 Quick Start

### Live Demo
**Try the application online**: [Streamlit Cloud Demo](https://your-app-name.streamlit.app)

### Local Installation

#### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FacialEmotionRecognition.git
   cd FacialEmotionRecognition
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   # Windows
   run_app.bat

   # Or manually
   python -m streamlit run streamlit_app.py
   ```

5. **Open your browser** and go to `http://localhost:8501`

## 📱 Web Application

The system provides a modern web interface with two main modes:

### 🎥 Real-time Detection
- **Live webcam emotion recognition**
- **Interactive confidence visualization**
- **Real-time face detection and emotion classification**
- **Session statistics and emotion history**

### 📊 Results Demo
- **Comprehensive project results and analysis**
- **Model performance comparison**
- **Training statistics and visualizations**
- **Technical documentation**

## 🏗️ Project Structure

```
FacialEmotionRecognition/
├── web_app/                 # Streamlit web application
│   ├── app.py              # Real-time detection app
│   ├── main_app.py         # Main application launcher
│   └── templates/          # HTML templates
├── models/                 # Trained model files
├── data/                   # Dataset and training data
├── notebooks/              # Jupyter notebooks for training
├── docs/                   # Project documentation
├── tests/                  # Test scripts and utilities
├── scripts/                # Utility scripts
├── reports/                # Training reports and results
├── requirements.txt        # Python dependencies
└── streamlit_app.py       # Streamlit Cloud entry point
```

## 🧠 Model Information

### Best Performing Model
- **Model**: HPO Optimized CNN (Converted)
- **Parameters**: 12.4M
- **Input Size**: 48x48x3 (RGB)
- **Accuracy**: 79.69%
- **Classes**: 4 emotions (Happy, Neutral, Sad, Surprise)

### Available Models
- HPO Optimized CNN (12.4M parameters)
- Complex RGB CNN (79.69% accuracy)
- Complex GS192 Strong Regularized CNN (77.34% accuracy)
- Complex CNN (78.13% accuracy)
- Transfer Learning models (ResNet50, VGG16, DenseNet121)

## 🛠️ Technical Stack

- **Deep Learning**: TensorFlow/Keras
- **Web Framework**: Streamlit
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Development**: Jupyter Notebooks

## 📈 Performance

The system achieves state-of-the-art performance on facial emotion recognition:

- **Overall Accuracy**: 79.69%
- **Real-time Processing**: 30+ FPS
- **Model Size**: 12.4M parameters
- **Inference Time**: <50ms per frame

## 🚀 Deployment

### Local Development
```bash
python -m streamlit run streamlit_app.py
```

### Streamlit Cloud
The application is ready for deployment on Streamlit Cloud:
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy automatically

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:
- Training procedures and experiments
- Model architecture details
- Performance analysis
- API documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MIT Capstone Project
- TensorFlow/Keras community
- OpenCV contributors
- Streamlit team

---

**Status**: ✅ Production Ready
**Last Updated**: October 2025
**Version**: 1.0.0
