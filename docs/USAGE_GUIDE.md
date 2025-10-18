# Facial Emotion Recognition - Usage Guide

## 🚀 Quick Start

### Launch the Application
```bash
# Option 1: Double-click the batch file
run_app.bat

# Option 2: PowerShell script
.\run_app.ps1

# Option 3: Manual launch
.venv\Scripts\activate
cd web_app
python -m streamlit run main_app.py
```

## 📱 Application Modes

### 1. 📊 Results Demo (Recommended)
- **Purpose**: View comprehensive project results and analysis
- **Features**:
  - Interactive model comparison charts
  - Performance metrics visualization
  - Technical implementation details
  - Model efficiency analysis
- **Status**: ✅ Fully functional
- **Best for**: Understanding project achievements and model performance

### 2. 🎥 Real-time Detection
- **Purpose**: Live emotion recognition using webcam
- **Features**:
  - Real-time face detection
  - Emotion classification
  - Confidence visualization
  - Emotion tracking over time
- **Status**: ⚠️ Requires model compatibility
- **Best for**: Live emotion recognition demonstrations

### 3. ℹ️ About
- **Purpose**: Project information and technical details
- **Features**:
  - Project overview
  - Technical stack information
  - Model performance summary
  - Quick start guide
- **Status**: ✅ Fully functional
- **Best for**: Learning about the project

## 🔧 Troubleshooting

### Model Compatibility Issues
If you encounter model loading errors in Real-time Detection mode:

1. **Use Results Demo**: This mode is guaranteed to work and shows all project results
2. **Check Model Files**: Ensure model files exist in `../models/` directory
3. **Version Compatibility**: Models were trained with older Keras version

### Common Issues

#### Streamlit Not Found
```bash
# Solution: Activate virtual environment first
.venv\Scripts\activate
```

#### Import Errors
```bash
# Solution: Ensure you're in the correct directory
cd web_app
```

#### Model Loading Errors
- This is expected due to Keras version differences
- Use Results Demo mode for guaranteed functionality

## 📊 Understanding the Results

### Model Performance
- **Best Model**: Complex RGB CNN (79.69% accuracy)
- **Comparison**: 8 different architectures tested
- **Metrics**: Accuracy, F1-Score, Training Time, Parameters

### Key Insights
- Custom CNNs outperform transfer learning
- RGB models work better than grayscale
- Model complexity affects both accuracy and training time

## 🎯 Best Practices

1. **Start with Results Demo**: Always begin with the results demo to understand the project
2. **Check Compatibility**: Real-time detection may have model loading issues
3. **Use Virtual Environment**: Always activate the virtual environment before running
4. **Browser Compatibility**: Use modern browsers (Chrome, Firefox, Edge)

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Verify virtual environment is activated
3. Ensure all dependencies are installed
4. Use Results Demo mode for guaranteed functionality

## 🎉 Enjoy Exploring!

The application provides comprehensive insights into facial emotion recognition using deep learning. Start with the Results Demo to see the full project achievements!
