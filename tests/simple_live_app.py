"""
Simple Live Emotion Recognition App
"""

import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Page configuration is handled in streamlit_app.py

# Emotion labels
EMOTION_LABELS = ["happy", "neutral", "sad", "surprise"]


def load_emotion_model():
    """Load the converted emotion recognition model"""
    try:
        model_path = "../models/converted_best_hpo_optimized.keras"
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.success(f"✅ Model loaded: {model_path}")
            return model
        else:
            st.error(f"❌ Model not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def detect_face(frame):
    """Detect face in frame using Haar Cascade"""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face_roi = frame[y : y + h, x : x + w]
            return face_roi, (x, y, w, h)
        return None, None
    except Exception as e:
        st.warning(f"Face detection error: {str(e)}")
        return None, None


def preprocess_face(face_roi):
    """Preprocess face for model input"""
    try:
        # Resize to 48x48
        face_resized = cv2.resize(face_roi, (48, 48))
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        # Normalize
        face_normalized = face_rgb.astype(np.float32) / 255.0
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        return face_batch
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None


def main():
    st.title("😊 Simple Live Emotion Recognition")
    st.markdown("**Real-time emotion detection using your trained model**")

    # Load model
    model = load_emotion_model()
    if model is None:
        st.stop()

    # Initialize session state
    if "video_running" not in st.session_state:
        st.session_state.video_running = False

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Camera Feed")

        # Start/Stop button
        if st.button(
            "🎥 Start Live Video"
            if not st.session_state.video_running
            else "⏹️ Stop Video"
        ):
            st.session_state.video_running = not st.session_state.video_running
            st.rerun()

        if st.session_state.video_running:
            st.info("🎥 Live video is active!")

            # Camera input
            camera_input = st.camera_input("Look at the camera for emotion detection")

            if camera_input is not None:
                try:
                    # Convert to OpenCV format
                    image = Image.open(camera_input)
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # Detect face
                    face_roi, face_coords = detect_face(frame)

                    if face_roi is not None and face_coords is not None:
                        # Draw rectangle around face
                        x, y, w, h = face_coords
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Preprocess face
                        processed_face = preprocess_face(face_roi)

                        if processed_face is not None:
                            # Make prediction
                            predictions = model.predict(processed_face, verbose=0)
                            if len(predictions) > 0:
                                predictions = predictions[0]
                                confidence_scores = predictions * 100

                                # Find best emotion
                                best_emotion_idx = np.argmax(confidence_scores)
                                best_emotion = EMOTION_LABELS[best_emotion_idx]
                                best_confidence = confidence_scores[best_emotion_idx]

                                # Draw emotion text on frame
                                cv2.putText(
                                    frame,
                                    f"{best_emotion}: {best_confidence:.1f}%",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )

                                # Display results
                                with col2:
                                    st.subheader("Current Prediction")
                                    st.markdown(f"### {best_emotion.capitalize()} 😊")
                                    st.markdown(
                                        f"**Confidence: {best_confidence:.1f}%**"
                                    )

                                    # Show all emotions
                                    st.subheader("All Emotions")
                                    for i, emotion in enumerate(EMOTION_LABELS):
                                        confidence = confidence_scores[i]
                                        st.markdown(
                                            f"**{emotion.capitalize()}:** {confidence:.1f}%"
                                        )
                                        st.progress(confidence / 100.0)

                    # Convert frame back to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(
                        frame_rgb,
                        channels="RGB",
                        use_column_width=True,
                        caption="Processed frame with emotion detection",
                    )

                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
        else:
            st.info("Click 'Start Live Video' to begin emotion detection")

    with col2:
        st.subheader("Model Information")
        st.write(f"**Input Shape:** {model.input_shape}")
        st.write(f"**Parameters:** {model.count_params():,}")
        st.write(f"**Emotions:** {', '.join(EMOTION_LABELS)}")


if __name__ == "__main__":
    main()
