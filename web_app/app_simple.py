"""
Facial Emotion Recognition App with WebRTC
Full-featured real-time emotion detection using streamlit-webrtc
"""

import logging
import os
import threading
import time
import warnings
from collections import deque

import av
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
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

# Global model cache for WebRTC processor
_MODEL = None
_MODEL_PATH = None

# Debug mode toggle - moved to main area to ensure visibility
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = True


def ensure_model_available():
    """Ensure model is available, download if needed (lazy loading)"""
    # Check if model is already downloaded
    possible_paths = [
        "models/converted_best_hpo_optimized.keras",  # Local development
        "converted_best_hpo_optimized.keras",  # Streamlit Cloud root
        os.path.join(
            os.getcwd(), "converted_best_hpo_optimized.keras"
        ),  # Absolute path
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path

    # Model not found, download it
    print("DEBUG: Model not found locally, downloading...")
    try:
        if download_model_from_dropbox():
            # Try to find the downloaded model
            for path in possible_paths:
                if os.path.exists(path):
                    return path
    except Exception as e:
        print(f"DEBUG: Error downloading model: {e}")

    return None


def get_model_once(model_path):
    """Simple thread-safe model cache without Streamlit dependencies"""
    global _MODEL, _MODEL_PATH
    if model_path is None:
        return None
    if _MODEL is None or _MODEL_PATH != model_path:
        _MODEL = load_model(model_path, compile=False)
        _MODEL_PATH = model_path
    return _MODEL


class WebRTCEmotionProcessor(VideoProcessorBase):
    """WebRTC processor for real-time emotion detection"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.emotion_labels = EMOTION_LABELS
        self.model_input_size = (48, 48)
        self.recent_predictions = deque(maxlen=5)
        self.last_update = 0.0
        self.update_interval = 0.2  # More frequent updates for real-time feel
        self.current_emotion = None
        self.current_confidence = None
        self.current_scores = None
        self.last_scores = None  # Keep last good scores

        # Threading for data synchronization
        self._lock = threading.Lock()
        self.ready = threading.Event()
        if self.model_loaded:
            self.ready.set()

        # Caching for stable overlay rendering
        self.last_bbox = None  # (x,y,w,h)
        self.last_text = None  # "emotion: 95.1%"
        self.hold_ms = 2000  # hold last overlay for 2.0s
        self.last_seen = 0.0
        self.hold_s = 0.8  # Hold scores for 0.8 seconds

        # Debug logging throttling
        self._last_log_ts = 0.0
        self.model_id = id(self.model)

        # Recognition state
        self.recognition_active = False

    def load_model_once(self):
        """Load model if not already loaded"""
        if not self.model_loaded and self.model_path:
            try:
                # Load model with timeout to prevent blocking
                import threading

                def load_model_thread():
                    try:
                        self.model = load_model(self.model_path, compile=False)
                        self.model_loaded = True
                        self.ready.set()
                        print("DEBUG: Model loaded successfully in processor")
                    except Exception as e:
                        print(f"DEBUG: Error loading model in processor: {e}")

                # Start loading in background thread
                if (
                    not hasattr(self, "_loading_thread")
                    or not self._loading_thread.is_alive()
                ):
                    self._loading_thread = threading.Thread(
                        target=load_model_thread, daemon=True
                    )
                    self._loading_thread.start()

                return self.model_loaded
            except Exception as e:
                print(f"DEBUG: Error starting model load: {e}")
                return False
        return self.model_loaded

    def preprocess_face(self, face, target_size):
        face_resized = cv2.resize(face, target_size)
        if len(face_resized.shape) == 3:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        face_norm = face_rgb.astype(np.float32) / 255.0
        return np.expand_dims(face_norm, axis=0)

    def smooth_predictions(self, new_prediction):
        self.recent_predictions.append(new_prediction)
        if len(self.recent_predictions) < 3:
            return new_prediction
        return np.mean(list(self.recent_predictions), axis=0)

    def _draw_overlay(self, img):
        """Draw cached overlay on every frame to prevent flickering"""
        # Only draw overlay if model is loaded, recognition is active, and we have recent data
        if (
            self.model_loaded
            and self.recognition_active
            and self.last_bbox
            and (time.time() - self.last_seen) * 1000 < self.hold_ms
        ):
            x, y, w, h = self.last_bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if self.last_text:
                cv2.putText(
                    img,
                    self.last_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                # Debug: show model ID
                if st.session_state.debug_mode:
                    cv2.putText(
                        img,
                        f"id:{self.model_id}",
                        (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Try to load model, but don't block if it's still loading
        self.load_model_once()

        # If model is not loaded yet, just return the frame without processing
        if not self.model_loaded:
            return frame
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        if now - self.last_update >= self.update_interval:
            self.last_update = now
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
            )  # More stable face detection parameters
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                roi = img[y : y + h, x : x + w]
                sample = self.preprocess_face(roi, self.model_input_size)
                preds = self.model.predict(sample, verbose=0)[0] * 100
                scores = self.smooth_predictions(preds)
                k = int(np.argmax(scores))

                # Update with lock to prevent race conditions
                with self._lock:
                    self.current_emotion = self.emotion_labels[k]
                    self.current_confidence = float(scores[k])
                    self.current_scores = scores
                    self.last_scores = scores  # Remember last good scores
                    self.last_bbox = (x, y, w, h)
                    self.last_text = (
                        f"{self.current_emotion}: {self.current_confidence:.1f}%"
                    )
                    self.last_seen = now

                # Debug logging (throttled to 1 Hz)
                if (time.time() - self._last_log_ts) > 1.0:
                    print(
                        f"[PROC] face k={k} {self.current_emotion} conf={self.current_confidence:.1f} scores={np.round(scores,1)}"
                    )
                    self._last_log_ts = time.time()
            else:
                # Never clear results immediately. Hold last valid values, clear only if face missing for long time (>2.5s)
                age = now - self.last_seen  # сколько времени мы не видели лицо
                with self._lock:
                    if (
                        hasattr(self, "last_scores")
                        and self.last_scores is not None
                        and age < max(self.hold_s, 2.5)
                    ):
                        # держим последние валидные результаты
                        self.current_scores = self.last_scores
                        k = int(np.argmax(self.current_scores))
                        self.current_emotion = self.emotion_labels[k]
                        self.current_confidence = float(self.current_scores[k])
                    else:
                        # очищаем ТОЛЬКО если лицо пропало заметно надолго
                        self.current_emotion = None
                        self.current_confidence = None
                        self.current_scores = None

                # Debug logging for no face case
                if (time.time() - self._last_log_ts) > 1.0:
                    print(
                        f"[PROC] no_face age={time.time()-self.last_seen:.2f}s last_scores={hasattr(self, 'last_scores') and self.last_scores is not None}"
                    )
                    self._last_log_ts = time.time()
                    # self.last_scores = None  # Keep for UI that updates later

        # Draw last known overlay on EVERY frame
        self._draw_overlay(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def snapshot(proc):
    """Safely read processor data with lock"""
    with proc._lock:
        scores = (
            proc.current_scores
            if proc.current_scores is not None
            else (proc.last_scores if hasattr(proc, "last_scores") else None)
        )
        return {
            "ready": proc.ready.is_set(),
            "loaded": proc.model_loaded,
            "emotion": proc.current_emotion,
            "conf": (
                float(proc.current_confidence)
                if proc.current_confidence is not None
                else None
            ),
            "scores": (
                None if scores is None else np.asarray(scores, dtype=float).tolist()
            ),
            "last_seen_age": (time.time() - proc.last_seen) if proc.last_seen else None,
        }


def is_recent(s, window=1.0):
    """Check if data is recent (within window seconds)"""
    a = s.get("last_seen_age")
    return (a is not None) and (a <= window)


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
            if os.path.exists(path) and os.path.getsize(path) > 0:
                model_path = path
                print(f"DEBUG: Model found at: {path}")
                break

        if model_path:
            return model_path
        else:
            # Try to download the model from Dropbox
            if download_model_from_dropbox():
                # Try to find the downloaded model
                for path in possible_paths:
                    if os.path.exists(path):
                        try:
                            test_model = load_model(path, compile=False)
                            if test_model is not None:
                                print(f"DEBUG: Downloaded model loaded at: {path}")
                                return path
                        except Exception as e:
                            print(f"Downloaded model at {path} failed to load: {e}")
                            continue
                st.error("Model downloaded but failed to load!")
                return None
            else:
                st.error("Best model not found and download failed!")
                return None

    except Exception as e:
        st.error(f"❌ Error loading emotion model: {str(e)}")
        return None


@st.cache_resource
def get_cached_model(model_path):
    """Cache the loaded model to avoid multiple loads"""
    try:
        print(f"DEBUG: Caching model from: {model_path}")
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Error caching model: {e}")
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
    # Debug mode toggle - always visible
    st.session_state.debug_mode = st.toggle(
        "🔧 Debug mode", value=st.session_state.debug_mode, key="debug_mode"
    )

    # Compact header
    st.markdown("#### Real-time Emotion Detection")

    # Initialize model path (will be loaded lazily when needed)
    model_path = None

    # Load model object for WebRTC processor (will be loaded when needed)
    model_obj = None

    # Debug: show model information
    if st.session_state.debug_mode:
        print("[MAIN] model_path=", model_path)
        print("[MAIN] model_id=", id(model_obj))
        if model_path:
            st.info(
                f"🧠 Model loaded: `{os.path.basename(model_path)}` (id={id(model_obj)})"
            )
        else:
            st.info("🧠 Model will be loaded when needed (lazy loading)")

    # Initialize session state
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = {
            emotion: deque(maxlen=50) for emotion in EMOTION_LABELS
        }

    if "video_running" not in st.session_state:
        st.session_state.video_running = False

    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = 50

    if "update_frequency" not in st.session_state:
        st.session_state.update_frequency = 1.0

    if "show_confidence_bars" not in st.session_state:
        st.session_state.show_confidence_bars = True

    if "show_history" not in st.session_state:
        st.session_state.show_history = True

    # Add smoothing for results
    if "recent_predictions" not in st.session_state:
        st.session_state.recent_predictions = deque(maxlen=5)  # Keep last 5 predictions

    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = 0

    # Sidebar with all controls
    with st.sidebar:
        # Video mode selection
        video_mode = st.radio(
            "Video Mode",
            ["WebRTC (Real-time)", "Camera Input (Photo)"],
            key="video_mode",
        )

        # Clear state when switching modes
        if "last_video_mode" not in st.session_state:
            st.session_state.last_video_mode = video_mode
        elif st.session_state.last_video_mode != video_mode:
            # Mode changed - clear all state
            st.session_state.video_running = False
            st.session_state.current_emotion = None
            st.session_state.current_confidence = None
            st.session_state.current_scores = None
            if "recent_predictions" in st.session_state:
                st.session_state.recent_predictions.clear()
            # Clear real-time history when switching modes
            if "rt_times" in st.session_state:
                st.session_state.rt_times.clear()
            if "rt_hist" in st.session_state:
                for emotion in st.session_state.rt_hist:
                    st.session_state.rt_hist[emotion].clear()
            st.session_state.last_video_mode = video_mode
            # Force rerun to clear UI
            st.rerun()

        # Video controls
        st.markdown("**Video Controls**")
        col1, col2 = st.columns(2)
        with col1:
            start = st.button("Start", key="start_video")
        with col2:
            stop = st.button("Stop", key="stop_video")

        if start:
            st.session_state.video_running = True
        if stop:
            st.session_state.video_running = False
            # Safely clear recent predictions
            if "recent_predictions" in st.session_state:
                if st.session_state.recent_predictions is not None:
                    try:
                        st.session_state.recent_predictions.clear()
                    except Exception:
                        st.session_state.recent_predictions = deque(maxlen=5)
                else:
                    st.session_state.recent_predictions = deque(maxlen=5)

        # Auto-stop video when switching to photo mode
        if video_mode == "Camera Input (Photo)" and st.session_state.video_running:
            st.session_state.video_running = False

        st.markdown("---")

        # Settings
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
                value=float(st.session_state.update_frequency),
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

    # Initialize real-time history tracking
    if "rt_times" not in st.session_state:
        st.session_state.rt_times = deque(maxlen=300)  # ~5 minutes at 1 Hz
        st.session_state.rt_hist = {e: deque(maxlen=300) for e in EMOTION_LABELS}
        st.session_state.last_hist_ts = 0.0

    # Video processing with WebRTC and Camera Input options
    if st.session_state.video_running:
        # Create two-column layout once - use more width
        col_camera, col_stats = st.columns([1, 1])

        # Fixed placeholders for status and video
        status_box = col_camera.empty()
        video_box = col_camera.container()

        if video_mode == "WebRTC (Real-time)":
            with video_box:
                st.caption("Live video (WebRTC)")

                # Lazy model loading - only when WebRTC mode is selected
                if model_path is None:
                    with st.spinner("Loading model..."):
                        model_path = ensure_model_available()
                        if model_path is None:
                            st.error("❌ Failed to load emotion recognition model!")
                            st.info(
                                "Please try again or switch to Model Analysis mode."
                            )
                            st.stop()

                # Load model object for WebRTC processor
                model_obj = get_model_once(model_path)

                # WebRTC Configuration with TURN server for NAT traversal
                rtc_configuration = RTCConfiguration(
                    {
                        "iceServers": [
                            {"urls": ["stun:stun.l.google.com:19302"]},
                            # TURN servers for NAT traversal (free tier)
                            {
                                "urls": ["turn:relay.metered.ca:80"],
                                "username": "free",
                                "credential": "free",
                            },
                            {
                                "urls": ["turn:relay.metered.ca:443"],
                                "username": "free",
                                "credential": "free",
                            },
                        ]
                    }
                )

                # WebRTC Streamer - fixed key for stability
                webrtc_ctx = webrtc_streamer(
                    key="emotion-detection",  # Fixed key for stability
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=lambda: WebRTCEmotionProcessor(model_path),
                    rtc_configuration=rtc_configuration,
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {
                                "ideal": 15
                            },  # Faster startup and more stable
                        },
                        "audio": False,
                    },
                    async_processing=False,  # Disable async to prevent freezing
                    desired_playing_state=st.session_state.video_running,
                    video_html_attrs={
                        "controls": True,
                        "style": {
                            "width": "100%",
                            "height": "auto",
                            "maxHeight": "400px",
                            "objectFit": "contain",
                        },
                    },
                )
                st.session_state.webrtc_ctx = webrtc_ctx

                # Автоматическое обновление UI когда распознавание активно
                is_playing = bool(webrtc_ctx and webrtc_ctx.state.playing)
                recognition_active = st.session_state.get("recognition_active", False)

                # Автоматическое обновление UI с помощью streamlit-autorefresh
                # ⛔️ удаляем/комментим блок с условным st.rerun — он даёт ровно один перезапуск
                # ✅ лёгкий автотик, пока видео играет и распознавание активно:
                if is_playing and st.session_state.get("recognition_active", False):
                    st_autorefresh(
                        interval=int(
                            1000 * max(0.5, float(st.session_state.update_frequency))
                        ),
                        key="rec_poll",  # стабильный ключ
                        limit=None,  # без ограничения
                    )
                    if st.session_state.debug_mode:
                        st.caption("🔄 Auto-refresh active")
                elif st.session_state.debug_mode:
                    if is_playing:
                        st.caption("⏸️ Video playing - recognition paused")
                    else:
                        st.caption("⏹️ Video stopped")

                # Status updates through fixed placeholder
                if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                    # Update recognition state in processor
                    webrtc_ctx.video_processor.recognition_active = (
                        st.session_state.get("recognition_active", False)
                    )
                    status_box.write(
                        "🎥 Camera access granted! Emotion detection is active."
                    )
                elif st.session_state.video_running:
                    status_box.write("🔄 Connecting...")
                else:
                    status_box.write("▶️ Press START to begin.")

        else:  # Camera Input mode - handled separately below
            with video_box:
                st.caption("Live video (Camera Input)")
                st.info("📷 Switch to 'Camera Input (Photo)' mode to take photos")

        with col_stats:
            st.caption("Results")

            # Recognition control buttons
            col_start, col_stop, col_status = st.columns([1, 1, 1])
            with col_start:
                if st.button(
                    "▶️ Start Recognition", type="primary", key="start_recognition"
                ):
                    st.session_state.recognition_active = True
                    st.success("Recognition started!")
                    st.rerun()  # Immediately trigger auto-refresh
            with col_stop:
                if st.button("⏹️ Stop Recognition", key="stop_recognition"):
                    st.session_state.recognition_active = False
                    st.info("Recognition stopped!")
            with col_status:
                # Status indicator
                if st.session_state.get("recognition_active", False):
                    st.success("🟢 Active")
                else:
                    st.info("⏸️ Paused")

            # Initialize recognition state if not exists
            if "recognition_active" not in st.session_state:
                st.session_state.recognition_active = False

            # Debug panel
            dbg_box = st.expander(
                "🔎 Debug panel", expanded=st.session_state.debug_mode
            )

            # Single placeholder for results with throttling
            results_ph = st.empty()
            history_ph = st.empty()  # placeholder for history graph
            bars_ph = st.empty()  # placeholder for current bars
            last_ui = st.session_state.get("last_ui", 0.0)

            # Clear placeholders when switching modes
            if st.session_state.get("last_video_mode") != video_mode:
                results_ph.empty()
                history_ph.empty()
                bars_ph.empty()
                # Clear camera input when switching to WebRTC
                if video_mode == "WebRTC (Real-time)":
                    st.session_state.camera_photo = None

            # Check if we're in WebRTC mode and get results from processor
            if video_mode == "WebRTC (Real-time)" and st.session_state.video_running:
                # ВСЕГДА сначала пробуем локальный контекст из текущего запуска,
                # и только если его нет – берём из session_state
                active_ctx = (
                    webrtc_ctx
                    if "webrtc_ctx" in locals()
                    else st.session_state.get("webrtc_ctx")
                )
                processor = getattr(active_ctx, "video_processor", None)

                # Debug information
                if st.session_state.debug_mode:
                    ctx_info = {
                        "playing": bool(webrtc_ctx and webrtc_ctx.state.playing),
                        "ctx_exists": webrtc_ctx is not None,
                        "processor_exists": processor is not None,
                        "recognition_active": st.session_state.get(
                            "recognition_active", False
                        ),
                    }
                    if processor is not None:
                        snap = snapshot(processor)
                        dbg_box.write("**ctx:**")
                        dbg_box.json(ctx_info)
                        dbg_box.write("**proc snapshot:**")
                        dbg_box.json(
                            {
                                "loaded": snap["loaded"],
                                "ready": snap["ready"],
                                "emotion": snap["emotion"],
                                "conf": snap["conf"],
                                "scores": (
                                    [float(x) for x in snap["scores"]]
                                    if snap["scores"] is not None
                                    else None
                                ),
                            }
                        )
                    else:
                        dbg_box.json(ctx_info)

                if processor is None:
                    results_ph.info("🔄 Connecting WebRTC…")
                elif not processor.model_loaded:
                    results_ph.info("⏳ Loading model…")
                elif not st.session_state.get("recognition_active", False):
                    results_ph.info(
                        "⏸️ Recognition paused. Click 'Start Recognition' to begin."
                    )
                else:
                    s = snapshot(processor)
                    if s["scores"] is not None and is_recent(s, 1.0):
                        now = time.time()
                        # Update main result display (throttled)
                        if now - last_ui > 0.1:
                            st.session_state.last_ui = now
                            with results_ph:
                                st.markdown(
                                    f"### **{(s['emotion'] or '').capitalize()}**"
                                )
                                st.markdown(f"**Confidence: {s['conf']:.1f}%**")
                    else:
                        results_ph.info("👤 No recent face (last > 1s).")

            # 1) Current bars (update more frequently for better responsiveness) - only when processor is available
            if video_mode == "WebRTC (Real-time)" and st.session_state.video_running:
                # ВСЕГДА сначала пробуем локальный контекст из текущего запуска,
                # и только если его нет – берём из session_state
                active_ctx = (
                    webrtc_ctx
                    if "webrtc_ctx" in locals()
                    else st.session_state.get("webrtc_ctx")
                )
                processor = getattr(active_ctx, "video_processor", None)

                if processor is not None and st.session_state.get(
                    "recognition_active", False
                ):
                    s = snapshot(processor)
                    # Current bars (every ~0.5s) - use snapshot scores
                    if show_confidence_bars and s["scores"] is not None:
                        if (
                            time.time() - st.session_state.get("last_bars_ts", 0.0)
                            > 0.2
                        ):
                            st.session_state["last_bars_ts"] = time.time()
                            with bars_ph:
                                fig = go.Figure(
                                    [
                                        go.Bar(
                                            x=EMOTION_LABELS,
                                            y=[float(s) for s in s["scores"]],
                                            text=[
                                                f"{float(s):.1f}%" for s in s["scores"]
                                            ],
                                            textposition="auto",
                                        )
                                    ]
                                )
                                fig.update_layout(
                                    height=220,
                                    margin=dict(l=0, r=0, t=20, b=20),
                                    showlegend=False,
                                    yaxis=dict(
                                        tickformat=".1f", title="Confidence (%)"
                                    ),
                                )
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )

                    # History (every ~1s) - use snapshot scores
                    if (
                        s["scores"] is not None
                        and time.time() - st.session_state.last_hist_ts > 0.5
                    ):
                        st.session_state.last_hist_ts = time.time()
                        st.session_state.rt_times.append(st.session_state.last_hist_ts)
                        for i, e in enumerate(EMOTION_LABELS):
                            st.session_state.rt_hist[e].append(float(s["scores"][i]))

                        if show_history:
                            with history_ph:
                                hist_fig = go.Figure()
                                # Convert timestamps to relative time (seconds from start)
                                start_time = (
                                    st.session_state.rt_times[0]
                                    if st.session_state.rt_times
                                    else 0
                                )
                                t = [
                                    (ts - start_time)
                                    for ts in st.session_state.rt_times
                                ]
                                for e in EMOTION_LABELS:
                                    hist_fig.add_scatter(
                                        x=t,
                                        y=list(st.session_state.rt_hist[e]),
                                        name=e,
                                        mode="lines",
                                    )
                                hist_fig.update_layout(
                                    height=280,
                                    margin=dict(l=0, r=0, t=20, b=20),
                                    legend=dict(orientation="h"),
                                    xaxis=dict(title="Time (seconds)"),
                                    yaxis=dict(tickformat=".3f", title="Probability"),
                                )
                                st.plotly_chart(
                                    hist_fig,
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )

            # UI auto-refresh is handled at the top of main() function - no need for additional rerun

            # Handle Camera Input mode results (regardless of video_running state)
            if video_mode == "Camera Input (Photo)":
                if (
                    hasattr(st.session_state, "current_emotion")
                    and st.session_state.current_emotion is not None
                ):
                    with results_ph:
                        st.markdown(
                            f"### **{st.session_state.current_emotion.capitalize()}**"
                        )
                        st.markdown(
                            f"**Confidence: {st.session_state.current_confidence:.1f}%**"
                        )

                        # Display all emotions
                        if (
                            show_confidence_bars
                            and st.session_state.current_scores is not None
                        ):
                            st.markdown("**All Emotions:**")
                            for i, emotion in enumerate(EMOTION_LABELS):
                                score = st.session_state.current_scores[i]

                                # Create progress bar
                                progress = float(
                                    score / 100.0
                                )  # Convert numpy.float32 to Python float
                                st.markdown(f"**{emotion.capitalize()}:** {score:.1f}%")
                                st.progress(progress)

                        # Update history
                        if show_history and st.session_state.current_scores is not None:
                            for i, emotion_label in enumerate(EMOTION_LABELS):
                                st.session_state.emotion_history[emotion_label].append(
                                    st.session_state.current_scores[i]
                                )

                            # Display history
                            if any(st.session_state.emotion_history.values()):
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
                else:
                    if video_mode == "WebRTC (Real-time)":
                        if st.session_state.video_running:
                            results_ph.info(
                                "🎥 Video is running but no emotion detected yet. Please position your face in front of the camera."
                            )
                        else:
                            results_ph.info(
                                "🎥 Start the video stream to see emotion detection results."
                            )
                    else:
                        results_ph.info(
                            "📷 Take a photo to see emotion detection results."
                        )

    # File uploader and camera input (always available when not in WebRTC mode)
    if video_mode == "Camera Input (Photo)":
        st.markdown("**Upload Image or Take Photo**")

        # Try camera input first (works on some platforms)
        camera_photo = st.camera_input("Take a photo with your camera")

        # File uploader as alternative
        uploaded_file = st.file_uploader(
            "Or upload an image file", type=["png", "jpg", "jpeg"]
        )

        # Use camera photo if available, otherwise use uploaded file
        image_to_process = camera_photo if camera_photo is not None else uploaded_file

        if image_to_process is not None:
            image_source = "camera" if camera_photo is not None else "upload"
            print(
                f"DEBUG: Image from {image_source}: {image_to_process.name if hasattr(image_to_process, 'name') else 'camera_photo'}"
            )

            # Create two-column layout for image processing
            col_img, col_results = st.columns([1, 1])

            with col_img:
                # Show image
                image = Image.open(image_to_process)
                print(f"DEBUG: Image opened, size: {image.size}")
                st.image(image, width=400)

            with col_results:
                # Process image using simple recognizer
                with st.spinner("Analyzing emotion..."):
                    try:
                        # Convert PIL to OpenCV format
                        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                        # Try to detect face first
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades
                            + "haarcascade_frontalface_default.xml"
                        )
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Try different parameters for better face detection
                        # For small images (like training data), use more sensitive parameters
                        faces = face_cascade.detectMultiScale(
                            gray, 1.05, 3, minSize=(20, 20)
                        )

                        if len(faces) > 0:
                            # Use largest face
                            largest_face = max(faces, key=lambda x: x[2] * x[3])
                            x, y, w, h = largest_face
                            face_roi = frame[y : y + h, x : x + w]

                            # Preprocess face - FIXED for RGB
                            face_resized = cv2.resize(face_roi, (48, 48))
                            if len(face_resized.shape) == 3:
                                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                            else:
                                face_rgb = cv2.cvtColor(
                                    face_resized, cv2.COLOR_GRAY2RGB
                                )
                            face_normalized = face_rgb.astype(np.float32) / 255.0
                            processed_face = np.expand_dims(face_normalized, axis=0)

                            # Load model and predict
                            if model_path is None:
                                # Try to get model path for photo mode
                                model_path = ensure_model_available()

                            if model_path is None:
                                st.error("Model not available. Please try again.")
                                return

                            model = get_cached_model(model_path)
                            if model is None:
                                st.error("Failed to load model. Please try again.")
                                return

                            predictions = model.predict(processed_face, verbose=0)[0]
                            confidence_scores = predictions * 100

                            best_emotion_idx = np.argmax(confidence_scores)
                            best_emotion = EMOTION_LABELS[best_emotion_idx]
                            best_confidence = confidence_scores[best_emotion_idx]

                            # Display results
                            st.markdown(f"### **{best_emotion.capitalize()}**")
                            st.markdown(f"**Confidence: {best_confidence:.1f}%**")

                            # Display all emotions
                            if show_confidence_bars:
                                # Convert numpy arrays to Python floats for progress bars
                                confidence_scores_float = [
                                    float(score) for score in confidence_scores
                                ]
                                confidence_fig = create_confidence_bars(
                                    confidence_scores_float
                                )
                                st.plotly_chart(
                                    confidence_fig,
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )

                            st.info("✅ Face detected and processed")

                        else:
                            # Fallback: if no face detected, try to process the entire image
                            # This is useful for training images (48x48) that might not be detected by face cascade
                            st.warning(
                                "⚠️ No face detected, trying to process entire image..."
                            )

                            # Load model and predict on entire image
                            if model_path is None:
                                # Try to get model path for photo mode
                                model_path = ensure_model_available()

                            if model_path is None:
                                st.error("Model not available. Please try again.")
                                return

                            model = get_cached_model(model_path)
                            if model is not None:
                                # Resize entire image to 48x48 and process
                                img_resized = cv2.resize(frame, (48, 48))
                                if len(img_resized.shape) == 3:
                                    img_rgb = cv2.cvtColor(
                                        img_resized, cv2.COLOR_BGR2RGB
                                    )
                                else:
                                    img_rgb = cv2.cvtColor(
                                        img_resized, cv2.COLOR_GRAY2RGB
                                    )
                                img_normalized = img_rgb.astype(np.float32) / 255.0
                                processed_img = np.expand_dims(img_normalized, axis=0)

                                predictions = model.predict(processed_img, verbose=0)[0]
                                confidence_scores = predictions * 100

                                best_emotion_idx = np.argmax(confidence_scores)
                                best_emotion = EMOTION_LABELS[best_emotion_idx]
                                best_confidence = confidence_scores[best_emotion_idx]

                                # Display results
                                st.markdown(f"### **{best_emotion.capitalize()}**")
                                st.markdown(f"**Confidence: {best_confidence:.1f}%**")

                                # Display all emotions
                                if show_confidence_bars:
                                    confidence_scores_float = [
                                        float(score) for score in confidence_scores
                                    ]
                                    confidence_fig = create_confidence_bars(
                                        confidence_scores_float
                                    )
                                    st.plotly_chart(
                                        confidence_fig,
                                        use_container_width=True,
                                        config={"displayModeBar": False},
                                    )

                                st.info("✅ Image processed without face detection")
                            else:
                                st.error("❌ Model not available")

                    except Exception as e:
                        st.error(f"Error processing image: {e}")

    # Technical details (only show when not in video mode)
    if not st.session_state.video_running:
        with st.expander("Model Info"):
            st.markdown(
                f"""
            **HPO Optimized CNN** | 12.4M params | (48, 48, 3) | Threshold: {confidence_threshold}%
            """
            )


if __name__ == "__main__":
    main()
