#!/usr/bin/env python3
"""
Test with correct preprocessing matching training data
"""

import os
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def preprocess_for_training(image_path, target_size=(48, 48)):
    """Preprocess image to match training data format"""
    try:
        # Load image using keras preprocessing
        img = image.load_img(image_path, target_size=target_size, color_mode="rgb")
        img_array = image.img_to_array(img)

        # Normalize to [-1, 1] range (common for training)
        img_normalized = img_array / 127.5 - 1.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch
    except Exception as e:
        print(f"ERROR: Error preprocessing image: {e}")
        return None


def preprocess_standard(image_path, target_size=(48, 48)):
    """Standard preprocessing [0, 1] range"""
    try:
        img = image.load_img(image_path, target_size=target_size, color_mode="rgb")
        img_array = image.img_to_array(img)

        # Normalize to [0, 1] range
        img_normalized = img_array / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch
    except Exception as e:
        print(f"ERROR: Error preprocessing image: {e}")
        return None


def test_preprocessing_methods():
    """Test different preprocessing methods"""
    print("Testing different preprocessing methods...")
    print("=" * 60)

    # Load model
    model_path = os.path.join(project_root, "models/converted_best_hpo_optimized.keras")
    model = load_model(model_path)
    print(f"Loaded model: {model_path}")

    # Test images
    test_images = [
        ("data/test/happy/15705.jpg", "happy"),
        ("data/test/neutral/7205.jpg", "neutral"),
        ("data/test/sad/6798.jpg", "sad"),
        ("data/test/surprise/128.jpg", "surprise"),
    ]

    preprocessing_methods = [
        ("Standard [0,1]", preprocess_standard),
        ("Training [-1,1]", preprocess_for_training),
    ]

    emotion_labels = ["happy", "neutral", "sad", "surprise"]

    for method_name, preprocess_func in preprocessing_methods:
        print(f"\n--- Testing: {method_name} ---")

        correct_predictions = 0
        total_predictions = 0

        for image_path, expected_emotion in test_images:
            full_path = os.path.join(project_root, image_path)

            if not os.path.exists(full_path):
                print(f"ERROR: Image not found: {full_path}")
                continue

            # Preprocess image
            processed_img = preprocess_func(full_path)
            if processed_img is None:
                continue

            try:
                # Make prediction
                predictions = model.predict(processed_img, verbose=0)
                confidence_scores = predictions[0] * 100

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

            except Exception as e:
                print(f"   ERROR: Prediction failed: {e}")

        accuracy = (
            (correct_predictions / total_predictions * 100)
            if total_predictions > 0
            else 0
        )
        print(
            f"   Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)"
        )


def test_model_input_range():
    """Test what input range the model expects"""
    print("\n" + "=" * 60)
    print("Testing model input range...")

    # Load model
    model_path = os.path.join(project_root, "models/converted_best_hpo_optimized.keras")
    model = load_model(model_path)

    # Create test inputs with different ranges
    test_inputs = [
        ("Zeros", np.zeros((1, 48, 48, 3))),
        ("Ones", np.ones((1, 48, 48, 3))),
        ("Random [0,1]", np.random.random((1, 48, 48, 3))),
        ("Random [-1,1]", np.random.random((1, 48, 48, 3)) * 2 - 1),
        ("Random [0,255]", np.random.random((1, 48, 48, 3)) * 255),
    ]

    emotion_labels = ["happy", "neutral", "sad", "surprise"]

    for name, test_input in test_inputs:
        try:
            predictions = model.predict(test_input, verbose=0)
            confidence_scores = predictions[0] * 100

            best_idx = np.argmax(confidence_scores)
            predicted_emotion = emotion_labels[best_idx]
            best_confidence = confidence_scores[best_idx]

            print(f"{name:15}: {predicted_emotion} ({best_confidence:.1f}%)")

        except Exception as e:
            print(f"{name:15}: ERROR - {e}")


def main():
    """Main test function"""
    print("Facial Emotion Recognition - Correct Preprocessing Test")
    print("=" * 70)

    test_preprocessing_methods()
    test_model_input_range()

    print("\n" + "=" * 70)
    print("Preprocessing test completed!")


if __name__ == "__main__":
    main()
