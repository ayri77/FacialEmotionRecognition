#!/usr/bin/env python3
"""
Test with correct class labels matching the training data
"""

import os
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def test_with_correct_labels():
    """Test with correct class labels from training"""
    print("Testing with correct class labels...")
    print("=" * 60)

    # Load model
    model_path = os.path.join(project_root, "models/converted_best_hpo_optimized.keras")
    model = load_model(model_path)
    print(f"Loaded model: {model_path}")

    # Correct class order from training (from notebook)
    # Class indices: {'happy': 0, 'neutral': 1, 'sad': 2, 'surprise': 3}
    emotion_labels = ["happy", "neutral", "sad", "surprise"]
    print(f"Class order: {emotion_labels}")

    # Test multiple images from each class
    test_cases = [
        # (path, expected_class, expected_index)
        ("data/test/happy/15705.jpg", "happy", 0),
        ("data/test/happy/15707.jpg", "happy", 0),
        ("data/test/neutral/7205.jpg", "neutral", 1),
        ("data/test/neutral/7206.jpg", "neutral", 1),
        ("data/test/sad/6798.jpg", "sad", 2),
        ("data/test/sad/6799.jpg", "sad", 2),
        ("data/test/surprise/128.jpg", "surprise", 3),
        ("data/test/surprise/129.jpg", "surprise", 3),
    ]

    correct_predictions = 0
    total_predictions = 0

    for image_path, expected_emotion, expected_idx in test_cases:
        full_path = os.path.join(project_root, image_path)

        if not os.path.exists(full_path):
            print(f"ERROR: Image not found: {full_path}")
            continue

        print(
            f"\nTesting: {os.path.basename(image_path)} (expected: {expected_emotion}, idx: {expected_idx})"
        )

        # Preprocess image using keras preprocessing (same as training)
        try:
            img = image.load_img(full_path, target_size=(48, 48), color_mode="rgb")
            img_array = image.img_to_array(img)

            # Normalize to [0, 1] range
            img_normalized = img_array / 255.0

            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)

            # Make prediction
            predictions = model.predict(img_batch, verbose=0)
            confidence_scores = predictions[0] * 100

            # Get results
            best_idx = np.argmax(confidence_scores)
            predicted_emotion = emotion_labels[best_idx]
            best_confidence = confidence_scores[best_idx]

            is_correct = predicted_emotion == expected_emotion
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            status = "OK" if is_correct else "ERROR"
            print(
                f"   {status}: {expected_emotion} -> {predicted_emotion} ({best_confidence:.1f}%)"
            )
            print(f"   All scores: {dict(zip(emotion_labels, confidence_scores))}")

            # Check if the predicted index matches expected
            if best_idx == expected_idx:
                print(f"   Index match: {expected_idx} -> {best_idx} ✓")
            else:
                print(f"   Index mismatch: {expected_idx} -> {best_idx} ✗")

        except Exception as e:
            print(f"   ERROR: Processing failed: {e}")

    accuracy = (
        (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    )
    print(
        f"\nOverall accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)"
    )

    return accuracy


def test_class_distribution():
    """Test class distribution in predictions"""
    print("\n" + "=" * 60)
    print("Testing class distribution in predictions...")

    # Load model
    model_path = os.path.join(project_root, "models/converted_best_hpo_optimized.keras")
    model = load_model(model_path)

    emotion_labels = ["happy", "neutral", "sad", "surprise"]

    # Test on all available test images
    test_dirs = ["happy", "neutral", "sad", "surprise"]
    all_predictions = []

    for emotion_dir in test_dirs:
        emotion_path = os.path.join(project_root, "data", "test", emotion_dir)
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) if f.endswith(".jpg")]
            print(f"\nTesting {len(files)} images from {emotion_dir} class:")

            for i, filename in enumerate(files[:5]):  # Test first 5 images
                file_path = os.path.join(emotion_path, filename)

                try:
                    # Preprocess
                    img = image.load_img(
                        file_path, target_size=(48, 48), color_mode="rgb"
                    )
                    img_array = image.img_to_array(img)
                    img_normalized = img_array / 255.0
                    img_batch = np.expand_dims(img_normalized, axis=0)

                    # Predict
                    predictions = model.predict(img_batch, verbose=0)
                    confidence_scores = predictions[0] * 100

                    best_idx = np.argmax(confidence_scores)
                    predicted_emotion = emotion_labels[best_idx]
                    best_confidence = confidence_scores[best_idx]

                    all_predictions.append(predicted_emotion)

                    print(
                        f"   {filename}: {emotion_dir} -> {predicted_emotion} ({best_confidence:.1f}%)"
                    )

                except Exception as e:
                    print(f"   {filename}: ERROR - {e}")

    # Analyze distribution
    print("\nPrediction distribution:")
    for emotion in emotion_labels:
        count = all_predictions.count(emotion)
        percentage = (count / len(all_predictions) * 100) if all_predictions else 0
        print(f"   {emotion}: {count} ({percentage:.1f}%)")


def main():
    """Main test function"""
    print("Facial Emotion Recognition - Correct Labels Test")
    print("=" * 70)

    accuracy = test_with_correct_labels()
    test_class_distribution()

    print("\n" + "=" * 70)
    print("Correct labels test completed!")

    if accuracy >= 70:
        print("✅ Model shows good accuracy - ready for real-time integration!")
    else:
        print(
            "⚠️ Model accuracy is lower than expected - may need further investigation"
        )


if __name__ == "__main__":
    main()
