#!/usr/bin/env python3
"""
Test model loading and emotion recognition on static images
"""

import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def test_model_loading():
    """Test loading different models"""
    print("Testing model loading...")
    print("=" * 50)

    model_paths = [
        "models/converted_best_hpo_optimized.keras",
        "models/demo_model.keras",
        "models/best_complex_cnn_rgb.keras",
        "models/best_hpo_optimized.keras",
    ]

    loaded_models = []

    for model_path in model_paths:
        full_path = os.path.join(project_root, model_path)
        print(f"\nTrying to load: {model_path}")

        if not os.path.exists(full_path):
            print(f"ERROR: File not found: {full_path}")
            continue

        try:
            model = load_model(full_path)
            print(f"OK: Successfully loaded: {model_path}")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Parameters: {model.count_params():,}")
            loaded_models.append((model_path, model))
        except Exception as e:
            print(f"ERROR: Failed to load {model_path}: {str(e)[:100]}...")

    return loaded_models


def preprocess_image(image_path, target_size=(48, 48)):
    """Preprocess image for model input"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image: {image_path}")
            return None

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img_resized = cv2.resize(img_rgb, target_size)

        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch
    except Exception as e:
        print(f"ERROR: Error preprocessing image: {e}")
        return None


def test_emotion_recognition(
    model, image_path, emotion_labels=["happy", "neutral", "sad", "surprise"]
):
    """Test emotion recognition on a single image"""
    print(f"\nTesting emotion recognition on: {os.path.basename(image_path)}")

    # Preprocess image
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return None

    try:
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        confidence_scores = predictions[0] * 100

        # Get results
        results = {}
        for i, emotion in enumerate(emotion_labels):
            results[emotion] = confidence_scores[i]

        # Find best prediction
        best_emotion_idx = np.argmax(confidence_scores)
        best_emotion = emotion_labels[best_emotion_idx]
        best_confidence = confidence_scores[best_emotion_idx]

        print(f"   Best prediction: {best_emotion} ({best_confidence:.1f}%)")
        print(f"   All scores: {dict(zip(emotion_labels, confidence_scores))}")

        return {
            "best_emotion": best_emotion,
            "best_confidence": best_confidence,
            "all_scores": results,
        }

    except Exception as e:
        print(f"ERROR: Error during prediction: {e}")
        return None


def test_on_sample_images(loaded_models):
    """Test models on sample images from each emotion class"""
    print("\n" + "=" * 50)
    print("Testing on sample images...")

    emotion_classes = ["happy", "neutral", "sad", "surprise"]
    test_images = {}

    # Get one sample image from each class
    for emotion in emotion_classes:
        emotion_dir = os.path.join(project_root, "data", "test", emotion)
        if os.path.exists(emotion_dir):
            files = [f for f in os.listdir(emotion_dir) if f.endswith(".jpg")]
            if files:
                test_images[emotion] = os.path.join(emotion_dir, files[0])
                print(f"OK: Found test image for {emotion}: {files[0]}")
            else:
                print(f"ERROR: No images found for {emotion}")
        else:
            print(f"ERROR: Directory not found: {emotion_dir}")

    # Test each loaded model
    for model_path, model in loaded_models:
        print(f"\n--- Testing model: {model_path} ---")

        for emotion, image_path in test_images.items():
            result = test_emotion_recognition(model, image_path)
            if result:
                expected = emotion
                predicted = result["best_emotion"]
                confidence = result["best_confidence"]

                if expected == predicted:
                    print(f"   OK: {emotion}: Correct! ({confidence:.1f}%)")
                else:
                    print(
                        f"   ERROR: {emotion}: Expected {expected}, got {predicted} ({confidence:.1f}%)"
                    )


def main():
    """Main test function"""
    print("Facial Emotion Recognition - Model Testing")
    print("=" * 60)

    # Test model loading
    loaded_models = test_model_loading()

    if not loaded_models:
        print("\nERROR: No models could be loaded!")
        return

    print(f"\nOK: Successfully loaded {len(loaded_models)} model(s)")

    # Test on sample images
    test_on_sample_images(loaded_models)

    print("\n" + "=" * 60)
    print("Model testing completed!")


if __name__ == "__main__":
    main()
