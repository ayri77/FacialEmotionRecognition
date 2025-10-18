#!/usr/bin/env python3
"""
Test improved preprocessing for emotion recognition
"""

import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def preprocess_image_improved(image_path, target_size=(48, 48), grayscale=False):
    """Improved preprocessing with multiple approaches"""
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

        if grayscale:
            # Convert to grayscale
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            # Convert back to 3-channel for model compatibility
            img_processed = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        else:
            img_processed = img_resized

        # Normalize to [0, 1]
        img_normalized = img_processed.astype(np.float32) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch
    except Exception as e:
        print(f"ERROR: Error preprocessing image: {e}")
        return None


def test_different_preprocessing():
    """Test different preprocessing approaches"""
    print("Testing different preprocessing approaches...")
    print("=" * 60)

    # Load the best working model
    model_path = os.path.join(project_root, "models/converted_best_hpo_optimized.keras")
    model = load_model(model_path)
    print(f"Loaded model: {model_path}")
    print(f"Input shape: {model.input_shape}")

    # Test image
    test_image = os.path.join(project_root, "data/test/happy/15705.jpg")
    print(f"Test image: {os.path.basename(test_image)}")

    # Test different preprocessing approaches
    approaches = [
        ("RGB 48x48", False, (48, 48)),
        ("Grayscale 48x48", True, (48, 48)),
        ("RGB 192x192", False, (192, 192)),
        ("Grayscale 192x192", True, (192, 192)),
    ]

    for name, grayscale, size in approaches:
        print(f"\n--- Testing: {name} ---")

        # Preprocess image
        processed_img = preprocess_image_improved(test_image, size, grayscale)
        if processed_img is None:
            continue

        print(f"Processed image shape: {processed_img.shape}")

        try:
            # Make prediction
            predictions = model.predict(processed_img, verbose=0)
            confidence_scores = predictions[0] * 100

            emotion_labels = ["happy", "neutral", "sad", "surprise"]
            best_idx = np.argmax(confidence_scores)
            best_emotion = emotion_labels[best_idx]
            best_confidence = confidence_scores[best_idx]

            print(f"Best prediction: {best_emotion} ({best_confidence:.1f}%)")
            print(f"All scores: {dict(zip(emotion_labels, confidence_scores))}")

        except Exception as e:
            print(f"ERROR: Prediction failed: {e}")


def test_face_detection_preprocessing():
    """Test with face detection preprocessing"""
    print("\n" + "=" * 60)
    print("Testing with face detection preprocessing...")

    # Load model
    model_path = os.path.join(project_root, "models/converted_best_hpo_optimized.keras")
    model = load_model(model_path)

    # Load face cascade
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"Loaded face cascade: {cascade_path}")
    except:
        print("ERROR: Could not load face cascade")
        return

    # Test image
    test_image = os.path.join(project_root, "data/test/happy/15705.jpg")

    try:
        # Load and detect face
        img = cv2.imread(test_image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        print(f"Detected {len(faces)} faces")

        if len(faces) > 0:
            # Get largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face

            # Extract face region
            face_roi = img_rgb[y : y + h, x : x + w]

            # Resize face to model input size
            face_resized = cv2.resize(face_roi, (48, 48))

            # Normalize
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)

            print(f"Face region shape: {face_batch.shape}")

            # Make prediction
            predictions = model.predict(face_batch, verbose=0)
            confidence_scores = predictions[0] * 100

            emotion_labels = ["happy", "neutral", "sad", "surprise"]
            best_idx = np.argmax(confidence_scores)
            best_emotion = emotion_labels[best_idx]
            best_confidence = confidence_scores[best_idx]

            print(f"Best prediction: {best_emotion} ({best_confidence:.1f}%)")
            print(f"All scores: {dict(zip(emotion_labels, confidence_scores))}")

        else:
            print("No faces detected")

    except Exception as e:
        print(f"ERROR: Face detection failed: {e}")


def main():
    """Main test function"""
    print("Facial Emotion Recognition - Improved Preprocessing Test")
    print("=" * 70)

    test_different_preprocessing()
    test_face_detection_preprocessing()

    print("\n" + "=" * 70)
    print("Preprocessing test completed!")


if __name__ == "__main__":
    main()
