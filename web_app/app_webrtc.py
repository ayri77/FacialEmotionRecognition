"""
Facial Emotion Recognition with WebRTC
Modern real-time emotion detection using streamlit-webrtc
"""

import logging
import os
import time
import warnings
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoTransformerBase, webrtc_streamer
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


class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.emotion_labels = EMOTION_LABELS
        self.model_input_size = (48, 48)
        self.recent_predictions = deque(maxlen=5)
        self.last_update = 0
        self.update_interval = 0.5  # Update every 0.5 seconds

    def load_model(self, model_path):
        """Load the emotion recognition model"""
        try:
            self.model = load_model(model_path, compile=False)
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_face(self, face, target_size):
        """Preprocess face for emotion recognition"""
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
            print(f"Error preprocessing face: {e}")
            return None

    def smooth_predictions(self, new_prediction):
        """Smooth predictions over time to reduce flickering"""
        self.recent_predictions.append(new_prediction)

        if len(self.recent_predictions) < 3:
            return new_prediction

        # Average recent predictions
        avg_prediction = np.mean(list(self.recent_predictions), axis=0)
        return avg_prediction

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Transform video frame with emotion detection"""
        if self.model is None:
            return frame

        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Check if it's time to update
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return frame

        self.last_update = current_time

        try:
            # Detect faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Use largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_roi = img[y : y + h, x : x + w]

                # Preprocess face
                processed_face = self.preprocess_face(face_roi, self.model_input_size)

                if processed_face is not None:
                    # Predict emotion
                    predictions = self.model.predict(processed_face, verbose=0)[0]
                    confidence_scores = predictions * 100

                    # Smooth predictions
                    smoothed_scores = self.smooth_predictions(confidence_scores)

                    # Get best emotion
                    best_emotion_idx = np.argmax(smoothed_scores)
                    best_emotion = self.emotion_labels[best_emotion_idx]
                    best_confidence = smoothed_scores[best_emotion_idx]

                    # Draw rectangle around face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw emotion text
                    text = f"{best_emotion}: {best_confidence:.1f}%"
                    cv2.putText(
                        img,
                        text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # Store prediction for display
                    self.current_emotion = best_emotion
                    self.current_confidence = best_confidence
                    self.current_scores = smoothed_scores

            else:
                # No face detected
                self.current_emotion = None
                self.current_confidence = None
                self.current_scores = None

        except Exception as e:
            print(f"Error in transform: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


@st.cache_resource
def load_emotion_model():
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
                return path

        st.error("Model not found! Please ensure the model file is available.")
        return None

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Facial Emotion Recognition - WebRTC", page_icon="😊", layout="wide"
    )

    st.title("Facial Emotion Recognition - Real-time WebRTC")
    st.markdown("**Modern real-time emotion detection using WebRTC**")

    # Load model
    model_path = load_emotion_model()
    if model_path is None:
        st.error("❌ Cannot load emotion recognition model!")
        st.stop()

    st.success("✅ Model loaded successfully!")

    # WebRTC Configuration
    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                # Add TURN server for production:
                # {"urls": ["turn:your.turn.server:3478"], "username": "user", "credential": "pass"}
            ]
        }
    )

    # Create two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Video Stream")

        # Create transformer instance
        transformer = EmotionTransformer()
        if not transformer.load_model(model_path):
            st.error("Failed to load model in transformer!")
            st.stop()

        # WebRTC Streamer
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            mode="recvonly",
            video_transformer_factory=lambda: transformer,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.video_transformer:
            st.info("🎥 Camera access granted! Emotion detection is active.")
        else:
            st.warning("⚠️ Camera access required for real-time detection.")

    with col2:
        st.subheader("Current Results")

        if webrtc_ctx.video_transformer and hasattr(
            webrtc_ctx.video_transformer, "current_emotion"
        ):
            transformer = webrtc_ctx.video_transformer

            if transformer.current_emotion is not None:
                # Display main result
                st.markdown(f"### **{transformer.current_emotion.capitalize()}**")
                st.markdown(f"**Confidence: {transformer.current_confidence:.1f}%**")

                # Display all emotions
                st.markdown("**All Emotions:**")
                for i, emotion in enumerate(EMOTION_LABELS):
                    score = transformer.current_scores[i]
                    # color = EMOTION_COLORS.get(emotion, "#808080")  # Not used in this version

                    # Create progress bar
                    progress = score / 100.0
                    st.markdown(f"**{emotion.capitalize()}:** {score:.1f}%")
                    st.progress(progress)

            else:
                st.info(
                    "👤 No face detected. Please position your face in front of the camera."
                )
        else:
            st.info("🎥 Start the video stream to see emotion detection results.")

    # Instructions
    st.markdown("---")
    st.markdown("### Instructions:")
    st.markdown(
        """
    1. **Click 'Start'** to begin video stream
    2. **Allow camera access** when prompted by browser
    3. **Position your face** in front of the camera
    4. **View real-time results** in the right panel

    **Features:**
    - ✅ Real-time emotion detection
    - ✅ Face detection and tracking
    - ✅ Smooth predictions (reduces flickering)
    - ✅ Works in browser (no local camera access needed)
    - ✅ Uses best model (79.2% accuracy)

    **Tips:**
    - Ensure good lighting
    - Keep face clearly visible
    - Works best with frontal face views
    """
    )

    # Technical info
    with st.expander("Technical Details"):
        st.markdown(
            """
        **Technology Stack:**
        - **WebRTC**: Real-time video streaming
        - **Streamlit-WebRTC**: Browser-based video processing
        - **OpenCV**: Face detection and image processing
        - **TensorFlow/Keras**: Emotion recognition model
        - **Haar Cascades**: Face detection

        **Model Information:**
        - **Architecture**: Complex RGB CNN (5 blocks)
        - **Accuracy**: 79.2% on test set
        - **Input Size**: 48x48 grayscale
        - **Classes**: 4 emotions (happy, neutral, sad, surprise)
        - **Parameters**: 8.6M
        """
        )


if __name__ == "__main__":
    main()
