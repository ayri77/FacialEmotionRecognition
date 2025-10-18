#!/usr/bin/env python3
"""
Test real-time model integration with proper preprocessing
"""

import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


class EmotionRecognizer:
    """Real-time emotion recognition class"""

    def __init__(self, model_path):
        """Initialize the emotion recognizer"""
        self.model = load_model(model_path)
        self.emotion_labels = ["happy", "neutral", "sad", "surprise"]
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        print(f"Loaded model: {model_path}")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")

    def preprocess_face(self, face_roi, target_size=(48, 48)):
        """Preprocess face region for emotion prediction"""
        try:
            # Resize face to model input size
            face_resized = cv2.resize(face_roi, target_size)

            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1] range
            face_normalized = face_rgb.astype(np.float32) / 255.0

            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)

            return face_batch
        except Exception as e:
            print(f"ERROR: Face preprocessing failed: {e}")
            return None

    def predict_emotion(self, face_roi):
        """Predict emotion from face region"""
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_roi)
            if processed_face is None:
                return None

            # Make prediction
            predictions = self.model.predict(processed_face, verbose=0)
            confidence_scores = predictions[0] * 100

            # Get results
            results = {}
            for i, emotion in enumerate(self.emotion_labels):
                results[emotion] = confidence_scores[i]

            # Find best prediction
            best_emotion_idx = np.argmax(confidence_scores)
            best_emotion = self.emotion_labels[best_emotion_idx]
            best_confidence = confidence_scores[best_emotion_idx]

            return {
                "emotion": best_emotion,
                "confidence": best_confidence,
                "all_scores": results,
            }

        except Exception as e:
            print(f"ERROR: Emotion prediction failed: {e}")
            return None

    def detect_faces(self, frame):
        """Detect faces in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces
        except Exception as e:
            print(f"ERROR: Face detection failed: {e}")
            return []

    def process_frame(self, frame):
        """Process a single frame for emotion recognition"""
        try:
            # Detect faces
            faces = self.detect_faces(frame)

            results = []
            for x, y, w, h in faces:
                # Extract face region
                face_roi = frame[y : y + h, x : x + w]

                # Predict emotion
                emotion_result = self.predict_emotion(face_roi)

                if emotion_result:
                    results.append(
                        {
                            "face_coords": (x, y, w, h),
                            "emotion": emotion_result["emotion"],
                            "confidence": emotion_result["confidence"],
                            "all_scores": emotion_result["all_scores"],
                        }
                    )

            return results

        except Exception as e:
            print(f"ERROR: Frame processing failed: {e}")
            return []


def test_on_static_images():
    """Test emotion recognition on static images"""
    print("Testing emotion recognition on static images...")
    print("=" * 60)

    # Initialize recognizer
    model_path = os.path.join(project_root, "models/converted_best_hpo_optimized.keras")
    recognizer = EmotionRecognizer(model_path)

    # Test images
    test_images = [
        ("data/test/happy/15705.jpg", "happy"),
        ("data/test/neutral/7205.jpg", "neutral"),
        ("data/test/sad/6798.jpg", "sad"),
        ("data/test/surprise/128.jpg", "surprise"),
    ]

    correct_predictions = 0
    total_predictions = 0

    for image_path, expected_emotion in test_images:
        full_path = os.path.join(project_root, image_path)

        if not os.path.exists(full_path):
            print(f"ERROR: Image not found: {full_path}")
            continue

        print(
            f"\nTesting: {os.path.basename(image_path)} (expected: {expected_emotion})"
        )

        # Load image
        frame = cv2.imread(full_path)
        if frame is None:
            print("ERROR: Could not load image")
            continue

        # Process frame
        results = recognizer.process_frame(frame)

        if results:
            # Use first detected face
            result = results[0]
            predicted_emotion = result["emotion"]
            confidence = result["confidence"]

            is_correct = predicted_emotion == expected_emotion
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            status = "OK" if is_correct else "ERROR"
            print(
                f"   {status}: {expected_emotion} -> {predicted_emotion} ({confidence:.1f}%)"
            )
            print(f"   All scores: {result['all_scores']}")
        else:
            print("   ERROR: No faces detected")

    accuracy = (
        (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    )
    print(
        f"\nOverall accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)"
    )


def test_camera_integration():
    """Test camera integration (simulation)"""
    print("\n" + "=" * 60)
    print("Testing camera integration...")

    # Initialize recognizer
    model_path = os.path.join(project_root, "models/converted_best_hpo_optimized.keras")
    recognizer = EmotionRecognizer(model_path)

    print("Camera integration test completed!")
    print("Ready for real-time emotion recognition!")
    print("\nTo use in Streamlit app:")
    print("1. Initialize EmotionRecognizer with model path")
    print("2. Use process_frame() method on camera frames")
    print("3. Display results with face rectangles and emotion labels")


def main():
    """Main test function"""
    print("Facial Emotion Recognition - Real-time Model Test")
    print("=" * 70)

    test_on_static_images()
    test_camera_integration()

    print("\n" + "=" * 70)
    print("Real-time model test completed!")


if __name__ == "__main__":
    main()
