"""
Minimal Facial Emotion Recognition App
This version avoids all potential recursion issues
"""

import logging
import os
import warnings

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

EMOTION_LABELS = ["happy", "neutral", "sad", "surprise"]
EMOTION_COLORS = {
    "happy": "#FFD700",
    "neutral": "#808080",
    "sad": "#4169E1",
    "surprise": "#FF6347",
}


class SimpleEmotionRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False)
        self.emotion_labels = EMOTION_LABELS
        self.model_input_size = (48, 48)

    def preprocess_face(self, face, target_size):
        try:
            # Resize to target size
            face_resized = cv2.resize(face, target_size)

            # Convert to grayscale if needed
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized

            # Normalize to [0, 1]
            face_normalized = face_gray.astype(np.float32) / 255.0

            # Add batch dimension
            return np.expand_dims(face_normalized, axis=0)
        except Exception as e:
            st.error(f"Error preprocessing face: {e}")
            return None

    def process_image(self, image):
        """Process a single image and return emotion prediction"""
        try:
            # Convert PIL to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Try to detect face first
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Use largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_roi = frame[y : y + h, x : x + w]

                processed_face = self.preprocess_face(face_roi, self.model_input_size)
                if processed_face is not None:
                    predictions = self.model.predict(processed_face, verbose=0)[0]
                    confidence_scores = predictions * 100

                    best_emotion_idx = np.argmax(confidence_scores)
                    best_emotion = self.emotion_labels[best_emotion_idx]
                    # best_confidence = confidence_scores[best_emotion_idx]  # Not used in this version

                    return best_emotion, confidence_scores, True
            else:
                # Fallback: process entire image if it's 48x48
                if frame.shape[:2] == (48, 48) or (
                    len(frame.shape) == 3 and frame.shape[:2] == (48, 48)
                ):
                    processed_face = self.preprocess_face(frame, self.model_input_size)
                    if processed_face is not None:
                        predictions = self.model.predict(processed_face, verbose=0)[0]
                        confidence_scores = predictions * 100

                        best_emotion_idx = np.argmax(confidence_scores)
                        best_emotion = self.emotion_labels[best_emotion_idx]
                        # best_confidence = confidence_scores[best_emotion_idx]  # Not used in this version

                        return best_emotion, confidence_scores, False

        except Exception as e:
            st.error(f"Error processing image: {e}")

        return None, None, False


@st.cache_resource
def load_model_simple():
    """Load the emotion recognition model"""
    try:
        # Try different possible paths
        possible_paths = [
            "models/converted_best_hpo_optimized.keras",
            "converted_best_hpo_optimized.keras",
            os.path.join(os.getcwd(), "converted_best_hpo_optimized.keras"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return SimpleEmotionRecognizer(path)

        st.error("Model not found! Please ensure the model file is available.")
        return None

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Facial Emotion Recognition - Minimal", page_icon="😊", layout="wide"
    )

    st.title("Facial Emotion Recognition - Minimal Version")
    st.markdown("**Simple, stable version without video processing**")

    # Load model
    recognizer = load_model_simple()
    if recognizer is None:
        st.error("❌ Cannot load emotion recognition model!")
        st.stop()

    st.success("✅ Model loaded successfully!")

    # Image upload
    st.header("Upload Image for Emotion Recognition")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        help="Upload an image containing a face for emotion recognition",
    )

    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Input Image")
            image = Image.open(uploaded_file)
            st.image(image, width=400)

        with col2:
            st.subheader("Emotion Recognition Results")

            # Process image
            with st.spinner("Analyzing emotion..."):
                best_emotion, confidence_scores, face_detected = (
                    recognizer.process_image(image)
                )

            if best_emotion is not None and confidence_scores is not None:
                # Display main result
                confidence = confidence_scores[EMOTION_LABELS.index(best_emotion)]
                st.markdown(f"### **{best_emotion.capitalize()}**")
                st.markdown(f"**Confidence: {confidence:.1f}%**")

                # Display all emotions
                st.markdown("**All Emotions:**")
                for i, emotion in enumerate(EMOTION_LABELS):
                    score = confidence_scores[i]
                    # color = EMOTION_COLORS.get(emotion, "#808080")  # Not used in this version

                    # Create progress bar
                    progress = score / 100.0
                    st.markdown(f"**{emotion.capitalize()}:** {score:.1f}%")
                    st.progress(progress)

                # Face detection info
                if face_detected:
                    st.info("✅ Face detected and processed")
                else:
                    st.warning("⚠️ No face detected, processed entire image")

            else:
                st.error("❌ Could not process image. Please try a different image.")

    # Instructions
    st.markdown("---")
    st.markdown("### Instructions:")
    st.markdown(
        """
    1. **Upload an image** containing a face
    2. **Wait for processing** - the model will analyze the emotion
    3. **View results** - see the predicted emotion and confidence scores

    **Tips:**
    - Use clear, well-lit images
    - Face should be clearly visible
    - Works best with frontal face views
    """
    )


if __name__ == "__main__":
    main()
