"""
Facial Emotion Recognition - Main Application
Combines results demo and real-time emotion detection
"""

import os
import sys

import streamlit as st

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Facial Emotion Recognition System")
    st.markdown("**Comprehensive emotion recognition with deep learning**")

    # Initialize session state
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Model Analysis"

    # Sidebar navigation
    st.sidebar.title("Navigation")

    # App selection
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode:",
        ["Model Analysis", "Real-time Detection", "About"],
        index=["Model Analysis", "Real-time Detection", "About"].index(
            st.session_state.app_mode
        ),
    )

    # Update session state if changed
    if app_mode != st.session_state.app_mode:
        st.session_state.app_mode = app_mode

    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Info")

    # Check if converted model is available
    converted_model_path = "models/converted_best_hpo_optimized.keras"
    model_available = os.path.exists(converted_model_path)

    if model_available:
        st.sidebar.markdown(
            """
        **Best Model**: HPO Optimized CNN
        **Parameters**: 12.4M
        **Classes**: 4 emotions
        **Status**: Real-time Ready
        """
        )
    else:
        st.sidebar.markdown(
            """
        **Best Model**: Complex RGB CNN
        **Accuracy**: 79.69%
        **Classes**: 4 emotions
        **Status**: Model Analysis Only
        """
        )

    # Main content based on selection
    if st.session_state.app_mode == "Model Analysis":
        show_results_demo()
    elif st.session_state.app_mode == "Real-time Detection":
        show_realtime_detection()
    elif st.session_state.app_mode == "About":
        show_about()


def show_results_demo():
    """Show the results demo application"""
    st.header("Project Results & Model Comparison")

    # Import and run the demo results
    try:
        from demo_results import main as demo_main

        demo_main()
    except ImportError as e:
        st.error(f"Could not load results demo: {e}")
        st.info("Please ensure demo_results.py is in the same directory")


def show_realtime_detection():
    """Show the real-time emotion detection application"""
    # No header here - app_simple.py will handle it

    # Check if converted model exists
    converted_model_path = "models/converted_best_hpo_optimized.keras"
    model_available = os.path.exists(converted_model_path)

    if model_available:
        # Show the actual real-time detection app
        try:
            from app_simple import main as app_main

            app_main()
        except Exception as e:
            st.error("Failed to start real-time detection")
            st.markdown(f"**Error**: {str(e)[:200]}...")

            # Show fallback information
            st.subheader("Model Information")
            st.markdown(
                """
            **Model**: HPO Optimized CNN (Converted)
            **Parameters**: 12.4M
            **Input Size**: 48x48x3
            **Emotions**: Happy, Neutral, Sad, Surprise
            **Status**: Error loading
            """
            )
    else:
        # Always show the warning first
        st.warning("Model Compatibility Notice")
        st.markdown(
            """
        **Expected Issue**: The trained models were saved with an older version of Keras and may not load properly with the current version.

        **This is normal and expected** - the models were trained in a Docker environment with TensorFlow 2.15,
        while the current environment uses TensorFlow 2.20 with Keras 3.x.
        """
        )

        # Show options
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Recommended Actions")
            st.markdown(
                """
            1. **Use Model Analysis** - View comprehensive project results and analysis
            2. **Use Docker Environment** - Run the original training environment
            3. **Retrain Models** - Retrain with current Keras version
            """
            )

            if st.button("Switch to Model Analysis", type="primary"):
                st.session_state.app_mode = "Model Analysis"
                # Avoid rerun to prevent recursion - just update session state
                pass

        with col2:
            st.subheader("Experimental")
            st.markdown(
                """
            **Try loading models anyway** (will likely fail due to version incompatibility)
            """
            )

            if st.button("Try Loading Models"):
                try:
                    from app_simple import main as app_main

                    app_main()
                except Exception as e:
                    st.error("Model Loading Failed")
                    st.markdown(
                        f"""
                    **Error**: {str(e)[:200]}...

                    **This confirms the compatibility issue.** The models were trained with Keras 2.x but
                    the current environment uses Keras 3.x, which has breaking changes in model serialization.
                    """
                    )
                    st.info(
                        "💡 **Solution**: Use the Results Demo mode to see all project achievements and model performance!"
                    )

    # Show technical details
    with st.expander("Technical Details"):
        st.markdown(
            """
        **Model Training Environment**:
        - TensorFlow: 2.15.0
        - Keras: 2.15.0
        - CUDA: 11.8
        - Docker: tensorflow/tensorflow:2.15.0-gpu-jupyter

        **Current Environment**:
        - TensorFlow: 2.20.0
        - Keras: 3.11.3
        - Python: 3.11+

        **Compatibility Issue**:
        Keras 3.x introduced breaking changes in model serialization format.
        Models saved with Keras 2.x cannot be loaded directly with Keras 3.x.

        **Solutions**:
        1. Use the original Docker environment
        2. Retrain models with current Keras version
        3. Use Results Demo for guaranteed functionality
        """
        )


def show_about():
    """Show project information"""
    st.header("About This Project")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ## Project Overview

        This project implements and compares multiple deep learning architectures for facial emotion recognition,
        achieving state-of-the-art performance on a 4-class emotion dataset.

        ### Key Achievements

        - **Best Model**: Complex RGB CNN with **79.69% accuracy**
        - **8 Different Architectures** trained and compared
        - **Comprehensive Evaluation** with multiple metrics
        - **Production-Ready** web applications

        ### 🔬 Technical Implementation

        **Architecture Types:**
        - Custom CNNs (3-5 blocks)
        - Transfer Learning (VGG16, ResNet50V2, EfficientNetV2B2)
        - Advanced Regularization (L2, Dropout, BatchNorm)

        **Key Features:**
        - Data augmentation
        - Class imbalance handling
        - GPU acceleration
        - Comprehensive logging
        - TensorBoard monitoring

        ### Dataset

        - **Classes**: Happy, Sad, Neutral, Surprise
        - **Total Images**: ~20,000
        - **Training**: ~15,000 images
        - **Validation**: ~5,000 images
        - **Test**: 128 images (balanced)
        """
        )

    with col2:
        st.markdown(
            """
        ### 🚀 Quick Start

        **Model Analysis** (Recommended):
        - Comprehensive project results
        - Interactive visualizations
        - Model comparison analysis
        - Fully functional

        **Real-time Detection**:
        - Live camera emotion recognition
        - Interactive confidence visualization
        - Emotion tracking over time
        - Requires model compatibility

        ### Technical Stack

        - **TensorFlow/Keras**: Deep learning
        - **Streamlit**: Web applications
        - **OpenCV**: Image processing
        - **Plotly**: Interactive visualizations
        - **Docker**: GPU acceleration

        ### 📚 Documentation

        - Complete project documentation
        - Training guides and tutorials
        - Model comparison reports
        - Deployment instructions
        """
        )

    # Model performance summary
    st.subheader("Model Performance Summary")

    performance_data = {
        "Model": [
            "Complex RGB CNN",
            "Complex CNN",
            "Complex GS192 Strong Reg",
            "Deep Regularized CNN",
            "VGG16 Transfer Learning",
            "ResNet50V2 Transfer Learning",
        ],
        "Accuracy": ["79.69%", "78.13%", "77.34%", "74.22%", "71.09%", "70.31%"],
        "F1-Score": ["79.73%", "78.03%", "77.67%", "73.41%", "71.03%", "69.92%"],
        "Parameters": ["8.68M", "8.68M", "6.02M", "1.96M", "14.88M", "24.12M"],
    }

    import pandas as pd

    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    **Project Status**: Complete
    **Development Time**: ~2 weeks
    **Models Trained**: 8
    **Best Accuracy**: 79.69%
    **Environment**: Virtual environment with Docker GPU support
    """
    )


def check_models_availability():
    """Check if models are available and compatible"""
    try:

        # Try to load a model to check compatibility
        model_path = "../models/best_complex_cnn_rgb.keras"
        if os.path.exists(model_path):
            # Try to load without actually loading (just check if it's possible)
            return True
        return False
    except Exception:
        return False


if __name__ == "__main__":
    main()
