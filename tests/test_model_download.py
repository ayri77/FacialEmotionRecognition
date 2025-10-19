#!/usr/bin/env python3
"""
Test script to verify model download and loading works correctly
"""

import os
import sys

import requests
from tensorflow.keras.models import load_model


def test_model_download():
    """Test downloading the model from Google Drive"""
    print("Testing model download...")

    model_path = "converted_best_hpo_optimized.keras"
    dropbox_url = "https://www.dropbox.com/scl/fi/sx5r2umettw1ul8a2jb20/converted_best_hpo_optimized.keras?rlkey=qj150k3qhvdsiy59m7aktgh3u&st=rlqyj1q5&dl=1"

    try:
        print(f"Downloading from: {dropbox_url}")
        response = requests.get(dropbox_url, stream=True)
        response.raise_for_status()

        print(f"Saving to: {model_path}")
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Check file size
        file_size = os.path.getsize(model_path)
        print(f"Download successful! File size: {file_size / (1024*1024):.1f} MB")

        # Check if it's HTML (Google Drive warning page)
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(200)
            if "Google Drive" in content or "virus" in content.lower():
                print("WARNING: Downloaded HTML page instead of model file!")
                print("Content preview:", content[:100])
                return False

        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def test_model_loading():
    """Test loading the downloaded model"""
    print("\nTesting model loading...")

    model_path = "converted_best_hpo_optimized.keras"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False

    try:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)

        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Model parameters: {model.count_params():,}")

        # Test prediction with dummy data
        import numpy as np

        dummy_input = np.random.random((1, 48, 48, 3))
        prediction = model.predict(dummy_input, verbose=0)
        print(f"Test prediction shape: {prediction.shape}")
        print(f"Test prediction: {prediction[0]}")

        return True

    except Exception as e:
        print(f"Model loading failed: {e}")
        return False


def test_emotion_recognizer():
    """Test the EmotionRecognizer class"""
    print("\nTesting EmotionRecognizer class...")

    try:
        # Add web_app to path
        sys.path.append("../web_app")
        from app_simple import EmotionRecognizer

        model_path = "converted_best_hpo_optimized.keras"
        recognizer = EmotionRecognizer(model_path)

        print("EmotionRecognizer created successfully!")
        print(f"Model input size: {recognizer.model_input_size}")
        print(f"Emotion labels: {recognizer.emotion_labels}")

        return True

    except Exception as e:
        print(f"EmotionRecognizer test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Starting model download and loading tests...\n")

    # Test 1: Download model
    download_success = test_model_download()

    if download_success:
        # Test 2: Load model
        loading_success = test_model_loading()

        if loading_success:
            # Test 3: Test EmotionRecognizer
            recognizer_success = test_emotion_recognizer()

            if recognizer_success:
                print("\nAll tests passed! Model is ready for deployment.")
            else:
                print("\nEmotionRecognizer test failed.")
        else:
            print("\nModel loading test failed.")
    else:
        print("\nModel download test failed.")

    # Cleanup
    if os.path.exists("converted_best_hpo_optimized.keras"):
        print("\nCleaning up test file...")
        os.remove("converted_best_hpo_optimized.keras")
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
