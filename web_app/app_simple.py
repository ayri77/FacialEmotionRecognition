"""
Simple Facial Emotion Recognition App
This version avoids caching issues and focuses on functionality
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings

warnings.filterwarnings("ignore")

# Suppress TensorFlow deprecation warnings
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

EMOTION_LABELS = ["happy", "neutral", "sad", "surprise"]
EMOTION_COLORS = {
    "happy": "#FFD700",
    "neutral": "#808080",
    "sad": "#4169E1",
    "surprise": "#FF6347",
}


class EmotionRecognizer:
    def __init__(self, model_path="models/converted_best_hpo_optimized.keras"):
        self.model = None
        self.face_cascade = None
        self.model_input_size = (48, 48)
        self.emotion_labels = EMOTION_LABELS
        self.emotion_colors = EMOTION_COLORS

        # Load model and cascade
        self._load_model(model_path)
        self._load_face_cascade()

        # Hide model info message - show only if needed
        # if self.model:
        #     st.info(f"📊 Model info: {self.model.count_params():,} parameters, Input: {self.model.input_shape}")

    def _load_model(self, model_path):
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                # Hide success message - model is loaded silently
                if self.model.input_shape:
                    self.model_input_size = self.model.input_shape[1:3]
            else:
                st.error(f"❌ Model file not found: {model_path}")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")

    def _load_face_cascade(self):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                st.error("❌ Could not load Haar cascade classifier.")
        except Exception as e:
            st.error(f"❌ Error loading face cascade: {str(e)}")

    def preprocess_face(self, face_roi, target_size=(48, 48)):
        """Preprocess face region for emotion prediction"""
        try:
            face_resized = cv2.resize(face_roi, target_size)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb / 255.0
            return np.expand_dims(face_normalized, axis=0)
        except Exception as e:
            st.error(f"Error preprocessing face: {e}")
            return None

    def process_frame(self, frame):
        """Detect faces, predict emotions, and draw on frame"""
        if self.model is None:
            return frame, None, None

        # Try face detection first
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_roi = frame[y : y + h, x : x + w]

                processed_face = self.preprocess_face(face_roi, self.model_input_size)
                if processed_face is not None:
                    predictions = self.model.predict(processed_face, verbose=0)[0]
                    confidence_scores = predictions * 100

                    best_emotion_idx = np.argmax(confidence_scores)
                    best_emotion = self.emotion_labels[best_emotion_idx]
                    best_confidence = confidence_scores[best_emotion_idx]

                    # Draw rectangle and text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{best_emotion}: {best_confidence:.1f}%",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    return frame, best_emotion, confidence_scores

        # Fallback: if no face detected or face detection failed,
        # try to process the entire frame (useful for 48x48 images)
        if frame.shape[:2] == (48, 48) or frame.shape[:2] == (48, 48, 3):
            # This might be a 48x48 face image
            processed_face = self.preprocess_face(frame, self.model_input_size)
            if processed_face is not None:
                predictions = self.model.predict(processed_face, verbose=0)[0]
                confidence_scores = predictions * 100

                best_emotion_idx = np.argmax(confidence_scores)
                best_emotion = self.emotion_labels[best_emotion_idx]
                best_confidence = confidence_scores[best_emotion_idx]

                # Draw text on frame
                cv2.putText(
                    frame,
                    f"{best_emotion}: {best_confidence:.1f}%",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                return frame, best_emotion, confidence_scores

        return frame, None, None


def download_model_from_dropbox():
    """Download the best model from Dropbox"""
    import requests

    # Use absolute path for Streamlit Cloud
    model_path = os.path.join(os.getcwd(), "converted_best_hpo_optimized.keras")
    dropbox_url = "https://www.dropbox.com/scl/fi/sx5r2umettw1ul8a2jb20/converted_best_hpo_optimized.keras?rlkey=qj150k3qhvdsiy59m7aktgh3u&st=rlqyj1q5&dl=1"

    try:
        # Download the model
        st.info("Downloading the best emotion recognition model (79.2% accuracy)...")
        response = requests.get(dropbox_url, stream=True)
        response.raise_for_status()

        # Save the model to current directory
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        st.success("Model downloaded successfully!")
        return True

    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        return False


@st.cache_resource
def load_emotion_model():
    """Load the best trained emotion recognition model (79.2% accuracy)"""
    try:
        # Try different possible paths for the model
        possible_paths = [
            "models/converted_best_hpo_optimized.keras",  # Local development
            "converted_best_hpo_optimized.keras",  # Streamlit Cloud root
            os.path.join(
                os.getcwd(), "converted_best_hpo_optimized.keras"
            ),  # Absolute path
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path:
            recognizer = EmotionRecognizer(model_path)
            return recognizer
        else:
            # Try to download the model from Dropbox
            if download_model_from_dropbox():
                # Try to find the downloaded model
                for path in possible_paths:
                    if os.path.exists(path):
                        recognizer = EmotionRecognizer(path)
                        return recognizer
                st.error("Model downloaded but not found!")
                return None
            else:
                st.error("Best model not found and download failed!")
                return None

    except Exception as e:
        st.error(f"❌ Error loading emotion model: {str(e)}")
        return None


def create_confidence_bars(
    confidence_scores, emotion_labels=EMOTION_LABELS, emotion_colors=EMOTION_COLORS
):
    """Create confidence bar chart"""
    fig = go.Figure(
        data=[
            go.Bar(
                x=emotion_labels,
                y=confidence_scores,
                marker_color=[
                    emotion_colors.get(emotion, "#808080") for emotion in emotion_labels
                ],
                text=[f"{score:.1f}%" for score in confidence_scores],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        height=200,  # Smaller height
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
    )

    return fig


def smooth_predictions(recent_predictions):
    """Smooth predictions by averaging recent results"""
    if not recent_predictions or len(recent_predictions) == 0:
        return None

    try:
        # Safely convert to list and average all recent predictions
        if hasattr(recent_predictions, "__iter__"):
            predictions_list = list(recent_predictions)
        else:
            return None

        if len(predictions_list) == 0:
            return None

        # Ensure all items are numpy arrays
        valid_predictions = []
        for pred in predictions_list:
            if pred is not None and hasattr(pred, "__len__"):
                valid_predictions.append(pred)

        if len(valid_predictions) == 0:
            return None

        avg_scores = np.mean(valid_predictions, axis=0)
        return avg_scores
    except Exception:
        # If there's any error, return None
        return None


def main():
    """Main application function"""
    # Compact header
    st.markdown("#### Real-time Emotion Detection")

    # Load model (hide loading messages)
    recognizer = load_emotion_model()
    if recognizer is None:
        st.error("❌ Failed to load or create emotion recognition model!")
        st.markdown(
            """
        **The application cannot run without a model.**

        **Options:**
        1. **Use Model Analysis mode** - View comprehensive project results without real-time detection
        2. **Download trained models** - See [MODEL_DOWNLOAD.md](../MODEL_DOWNLOAD.md) for instructions
        3. **Run locally with models** - Copy models to the `models/` directory

        **For demonstration purposes**, you can use the Model Analysis mode to see all project achievements and model performance.
        """
        )

        if st.button("← Back to Main Menu"):
            st.session_state.app_mode = "Model Analysis"
            st.rerun()

        st.stop()

    # Initialize session state
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = {
            emotion: deque(maxlen=50) for emotion in EMOTION_LABELS
        }

    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = 50

    if "show_confidence_bars" not in st.session_state:
        st.session_state.show_confidence_bars = True

    if "show_history" not in st.session_state:
        st.session_state.show_history = True

    if "video_running" not in st.session_state:
        st.session_state.video_running = False

    # Add smoothing for results
    if "recent_predictions" not in st.session_state:
        st.session_state.recent_predictions = deque(maxlen=5)  # Keep last 5 predictions

    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = 0

    if "update_frequency" not in st.session_state:
        st.session_state.update_frequency = 1.0  # seconds

    # Sidebar with all controls
    with st.sidebar:
        # Video controls at the top
        st.markdown("**Video Controls**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start", key="start_video"):
                st.session_state.video_running = True
                st.rerun()

        with col2:
            if st.button("Stop", key="stop_video"):
                # Safely clear video state
                try:
                    st.session_state.video_running = False
                    # Safely clear recent predictions to avoid pop() errors
                    if (
                        "recent_predictions" in st.session_state
                        and st.session_state.recent_predictions is not None
                    ):
                        # Convert to list and clear safely
                        try:
                            list(st.session_state.recent_predictions)
                            st.session_state.recent_predictions.clear()
                        except:
                            # If deque is corrupted, reinitialize it
                            st.session_state.recent_predictions = deque(maxlen=5)
                    else:
                        # Initialize if doesn't exist
                        st.session_state.recent_predictions = deque(maxlen=5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error stopping video: {e}")
                    st.session_state.video_running = False
                    st.rerun()

        st.markdown("---")

        # Settings below video controls
        st.header("Controls")

        with st.form("settings_form"):
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold (%)",
                min_value=0,
                max_value=100,
                value=st.session_state.confidence_threshold,
                help="Minimum confidence required to display emotion",
            )

            # Update frequency
            update_frequency = st.slider(
                "Update Frequency (seconds)",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.update_frequency,
                step=0.1,
                help="How often to update results display",
            )

            # Display options
            show_confidence_bars = st.checkbox(
                "Show Confidence Bars", value=st.session_state.show_confidence_bars
            )

            show_history = st.checkbox(
                "Show Emotion History", value=st.session_state.show_history
            )

            # Submit button
            submitted = st.form_submit_button("Apply Settings")

            if submitted:
                st.session_state.confidence_threshold = confidence_threshold
                st.session_state.update_frequency = update_frequency
                st.session_state.show_confidence_bars = show_confidence_bars
                st.session_state.show_history = show_history
                # Clear recent predictions safely
                if (
                    "recent_predictions" in st.session_state
                    and st.session_state.recent_predictions is not None
                ):
                    st.session_state.recent_predictions.clear()
                else:
                    st.session_state.recent_predictions = deque(maxlen=5)
                st.success("Settings updated!")

    # Use current values from session state
    confidence_threshold = st.session_state.confidence_threshold
    update_frequency = st.session_state.update_frequency
    show_confidence_bars = st.session_state.show_confidence_bars
    show_history = st.session_state.show_history

    # Video processing
    if st.session_state.video_running:
        # Create two-column layout once - use more width
        col_camera, col_stats = st.columns([1, 1])

        with col_camera:
            st.caption("Live video")
            video_placeholder = st.empty()

        with col_stats:
            st.caption("Results")
            results_placeholder = st.empty()

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open camera. Please check if camera is available.")
            st.session_state.video_running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        frame_count = 0
        while st.session_state.video_running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from camera")
                break

            frame_count += 1

            # Process frame
            frame, best_emotion, confidence_scores = recognizer.process_frame(frame)

            if best_emotion is not None and confidence_scores is not None:
                # Add to recent predictions for smoothing (safely)
                try:
                    if "recent_predictions" not in st.session_state:
                        st.session_state.recent_predictions = deque(maxlen=5)
                    st.session_state.recent_predictions.append(confidence_scores)
                except Exception:
                    # If there's an error, reinitialize the deque
                    st.session_state.recent_predictions = deque(maxlen=5)
                    try:
                        st.session_state.recent_predictions.append(confidence_scores)
                    except:
                        pass  # Skip if still failing

                # Update history
                for i, emotion_label in enumerate(EMOTION_LABELS):
                    st.session_state.emotion_history[emotion_label].append(
                        confidence_scores[i]
                    )

            # Update results display based on user-defined frequency
            current_time = time.time()
            if current_time - st.session_state.last_update_time > update_frequency:
                st.session_state.last_update_time = current_time

                # Use smoothed predictions
                try:
                    smoothed_scores = smooth_predictions(
                        st.session_state.recent_predictions
                    )
                except Exception:
                    smoothed_scores = None

                if smoothed_scores is not None:
                    # Find best emotion from smoothed scores
                    best_emotion_idx = np.argmax(smoothed_scores)
                    best_emotion_smoothed = EMOTION_LABELS[best_emotion_idx]
                    best_confidence_smoothed = smoothed_scores[best_emotion_idx]

                    with results_placeholder.container():
                        st.markdown(
                            f"**{best_emotion_smoothed.capitalize()}** {best_confidence_smoothed:.1f}%"
                        )

                        if show_confidence_bars:
                            confidence_fig = create_confidence_bars(smoothed_scores)
                            st.plotly_chart(
                                confidence_fig,
                                use_container_width=True,
                                config={"displayModeBar": False},
                            )

                        if show_history and any(
                            st.session_state.emotion_history.values()
                        ):
                            st.markdown("**Average Emotions:**")
                            avg_emotions = {}
                            for emotion in EMOTION_LABELS:
                                history = st.session_state.emotion_history[emotion]
                                if history:
                                    avg_emotions[emotion] = np.mean(list(history))
                                else:
                                    avg_emotions[emotion] = 0

                            cols = st.columns(4)
                            for i, (emotion, avg_score) in enumerate(
                                avg_emotions.items()
                            ):
                                with cols[i]:
                                    st.metric(
                                        label=emotion.capitalize(),
                                        value=f"{avg_score:.1f}%",
                                        delta=None,
                                    )

                            # Show emotion history chart
                            if any(
                                len(history) > 1
                                for history in st.session_state.emotion_history.values()
                            ):
                                st.markdown("**Emotion History:**")
                                history_fig = go.Figure()

                                for emotion in EMOTION_LABELS:
                                    history = st.session_state.emotion_history[emotion]
                                    if history:
                                        history_fig.add_trace(
                                            go.Scatter(
                                                y=list(history),
                                                mode="lines",
                                                name=emotion.capitalize(),
                                                line=dict(
                                                    color=EMOTION_COLORS.get(
                                                        emotion, "#808080"
                                                    )
                                                ),
                                            )
                                        )

                                history_fig.update_layout(
                                    title="",
                                    xaxis_title="Time",
                                    yaxis_title="Confidence (%)",
                                    height=150,
                                    showlegend=True,
                                    margin=dict(l=0, r=0, t=0, b=0),
                                )

                                st.plotly_chart(
                                    history_fig,
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )

            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", width=600)

            time.sleep(0.1)

        cap.release()

    else:
        # File uploader
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Create two-column layout for image processing
            col_img, col_results = st.columns([1, 1])

            with col_img:
                # Show uploaded image
                image = Image.open(uploaded_file)
                st.image(image, width=400)

            with col_results:
                # Process frame
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                frame, best_emotion, confidence_scores = recognizer.process_frame(frame)

                if best_emotion is not None and confidence_scores is not None:
                    st.markdown(
                        f"**{best_emotion.capitalize()}** {confidence_scores[EMOTION_LABELS.index(best_emotion)]:.1f}%"
                    )

                    if show_confidence_bars:
                        confidence_fig = create_confidence_bars(confidence_scores)
                        st.plotly_chart(
                            confidence_fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                else:
                    st.warning("No faces detected")

    # Technical details (only show when not in video mode)
    if not st.session_state.video_running:
        with st.expander("Model Info"):
            st.markdown(
                f"""
            **HPO Optimized CNN** | {recognizer.model.count_params():,} params | {recognizer.model_input_size} | Threshold: {confidence_threshold}%
            """
            )


if __name__ == "__main__":
    main()
