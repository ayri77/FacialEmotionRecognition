"""
Simple test script for the web application
"""

import os

import numpy as np
from tensorflow.keras.models import load_model

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")


def test_model_loading():
    """Test model loading"""
    print("🧪 Testing model loading...")

    # Model paths to try
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
                print("✅ Model loaded successfully!")
                print(f"📊 Model input shape: {model.input_shape}")
                print(f"📊 Model output shape: {model.output_shape}")
                print(f"📊 Number of parameters: {model.count_params():,}")

                # Test prediction with dummy data
                dummy_input = np.random.random((1, 48, 48, 3)).astype(np.float32)
                prediction = model.predict(dummy_input, verbose=0)
                print(f"🧠 Test prediction shape: {prediction.shape}")
                print(f"🧠 Test prediction values: {prediction[0]}")

                return model

            except Exception as e:
                print(f"❌ Error loading {model_path}: {e}")
                continue

    print("❌ No models could be loaded!")
    return None


def test_imports():
    """Test required imports"""
    print("\n📦 Testing imports...")

    try:
        import cv2

        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False

    try:
        import flask

        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("🚀 Starting Web App Tests")
    print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    # Test model loading
    model = test_model_loading()

    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY:")
    print(f"📦 Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"🧠 Model loading: {'✅ PASS' if model is not None else '❌ FAIL'}")

    if imports_ok and model is not None:
        print("\n🎉 All tests passed! The web app should work.")
        print("🚀 To run the app: python flask_app.py")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")


if __name__ == "__main__":
    main()
