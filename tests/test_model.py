"""
Test script to verify model loading and basic functionality
"""

import os
import sys

import numpy as np
from tensorflow.keras.models import load_model

# Add parent directory to path
sys.path.append("..")


def test_model_loading():
    """Test if we can load the best model"""
    print("🧪 Testing model loading...")

    # List of models to try in order of preference
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    MODELS = ROOT / "models"

    model_paths = [
        MODELS / "best_hpo_optimized.keras",
    ]

    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"📁 Trying to load: {model_path}")
                model = load_model(model_path)
                print("✅ Successfully loaded model!")
                print(f"📊 Model input shape: {model.input_shape}")
                print(f"📊 Model output shape: {model.output_shape}")
                print(f"📊 Number of parameters: {model.count_params():,}")

                # Test prediction with dummy data
                dummy_input = np.random.random((1, 48, 48, 3)).astype(np.float32)
                prediction = model.predict(dummy_input, verbose=0)
                print(f"🧠 Test prediction shape: {prediction.shape}")
                print(f"🧠 Test prediction values: {prediction[0]}")

                return model, model_path

            except Exception as e:
                print(f"❌ Error loading {model_path}: {e}")
                continue

    print("❌ No models could be loaded!")
    return None, None


def test_emotion_labels():
    """Test emotion labels"""
    print("\n🎭 Testing emotion labels...")

    emotion_labels = ["happy", "neutral", "sad", "surprise"]
    print(f"📝 Emotion labels: {emotion_labels}")

    # Test prediction interpretation
    dummy_prediction = np.array([0.1, 0.2, 0.3, 0.4])  # Example prediction
    best_emotion_idx = np.argmax(dummy_prediction)
    best_emotion = emotion_labels[best_emotion_idx]
    best_confidence = dummy_prediction[best_emotion_idx] * 100

    print(f"🎯 Example prediction: {dummy_prediction}")
    print(f"🏆 Best emotion: {best_emotion} ({best_confidence:.1f}%)")

    return emotion_labels


def test_opencv_import():
    """Test OpenCV import and basic functionality"""
    print("\n📷 Testing OpenCV...")

    try:
        import cv2

        print("✅ OpenCV imported successfully")
        print(f"📊 OpenCV version: {cv2.__version__}")

        # Test face cascade
        face_cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if os.path.exists(face_cascade_path):
            print("✅ Face cascade file found")
        else:
            print("❌ Face cascade file not found")

        return True

    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False


def test_streamlit_import():
    """Test Streamlit import"""
    print("\n🌐 Testing Streamlit...")

    try:
        import streamlit as st

        print("✅ Streamlit imported successfully")
        print(f"📊 Streamlit version: {st.__version__}")
        return True

    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False


def test_plotly_import():
    """Test Plotly import"""
    print("\n📊 Testing Plotly...")

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        print("✅ Plotly imported successfully")
        return True

    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Facial Emotion Recognition Web App Tests")
    print("=" * 60)

    # Test model loading
    model, model_path = test_model_loading()

    # Test emotion labels
    emotion_labels = test_emotion_labels()

    # Test imports
    opencv_ok = test_opencv_import()
    streamlit_ok = test_streamlit_import()
    plotly_ok = test_plotly_import()

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print(f"🧠 Model loading: {'✅ PASS' if model is not None else '❌ FAIL'}")
    print(f"📷 OpenCV: {'✅ PASS' if opencv_ok else '❌ FAIL'}")
    print(f"🌐 Streamlit: {'✅ PASS' if streamlit_ok else '❌ FAIL'}")
    print(f"📊 Plotly: {'✅ PASS' if plotly_ok else '❌ FAIL'}")

    if model is not None and opencv_ok and streamlit_ok and plotly_ok:
        print("\n🎉 All tests passed! The web app should work correctly.")
        print("🚀 To run the app: streamlit run app.py")
    else:
        print("\n⚠️ Some tests failed. Please check the requirements:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
