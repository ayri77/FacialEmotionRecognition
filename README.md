# Facial Emotion Recognition

End-to-end deep learning system for facial emotion classification with real-time inference and web-based deployment.

This project focuses on the full ML lifecycle: data preparation, model experimentation, hyperparameter optimization, evaluation, and deployment via a production-oriented web interface.

---

## Problem

Build a robust emotion classification system capable of:

* Detecting faces in real-time
* Classifying emotions from RGB input
* Maintaining low inference latency
* Supporting reproducible experimentation and model comparison
* Providing a deployable inference interface

Emotion classes:

* Happy
* Neutral
* Sad
* Surprise

---

## Architecture

### Model Design

Primary production model:

* Custom CNN architecture
* 12.4M parameters
* Input: 48×48×3 (RGB)
* Hyperparameter optimization applied

Additional experiments:

* Regularized CNN variants
* Transfer learning (ResNet50, VGG16, DenseNet121)
* Grayscale vs RGB comparisons
* Strong regularization experiments

---

## Training Pipeline

* Structured dataset preprocessing
* Data normalization and augmentation
* Hyperparameter optimization (HPO)
* Regularization strategies
* Validation tracking
* Model comparison framework
* Conversion pipeline for deployment-ready inference models

---

## Performance

Best performing model:

* Validation Accuracy: 79.69%
* Model Size: 12.4M parameters
* Inference Time: <50ms per frame
* Real-time processing: ~30 FPS (hardware dependent)

Note:
The focus of this project is architectural experimentation and deployment integration rather than leaderboard optimization.

---

## Deployment

The system includes a Streamlit-based web interface supporting:

* Real-time webcam emotion detection
* Confidence visualization
* Session statistics
* Model selection
* Performance diagnostics

Run locally:

```
git clone https://github.com/ayri77/FacialEmotionRecognition.git
cd FacialEmotionRecognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

---

## Technical Stack

* TensorFlow / Keras
* OpenCV
* NumPy / Pandas
* Streamlit
* Matplotlib / Seaborn / Plotly

---

## Engineering Highlights

* Modular training experiments
* Reproducible training configuration
* Model version separation
* Deployment-ready inference pipeline
* Structured project layout separating training and application layers

---

## Project Structure

```
FacialEmotionRecognition/
├── web_app/           # Inference and UI layer
├── models/            # Trained model artifacts
├── notebooks/         # Training and experimentation
├── scripts/           # Utility scripts
├── tests/             # Validation utilities
├── docs/              # Technical documentation
```

---

## Author

Pavlo Borysov
Hamburg, Germany
