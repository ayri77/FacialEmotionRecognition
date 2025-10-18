"""
Facial Emotion Recognition - Project Results Demo
Demonstrates the project results and model performance without loading actual models
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Facial Emotion Recognition - Results Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load results data
@st.cache_data
def load_results():
    """Load project results data"""
    try:
        # Load models summary
        models_df = pd.read_csv("../reports/models_summary.csv")
        return models_df
    except FileNotFoundError:
        # Create demo data if file not found
        demo_data = {
            "Model": [
                "CNN-1 (baseline)",
                "CNN-2 (Deep Regularized)",
                "VGG16-TL",
                "ResNet50V2-TL",
                "EfficientNetV2B2-TL",
                "Complex-CNN-5Blocks",
                "Complex-GS-CNN-5Blocks",
                "Complex-RGB-CNN-5Blocks",
            ],
            "Test_Accuracy": [
                0.65625,
                0.7421875,
                0.7109375,
                0.703125,
                0.3671875,
                0.78125,
                0.7578125,
                0.796875,
            ],
            "Test_Macro_F1": [
                0.6216,
                0.7341,
                0.7103,
                0.6992,
                0.2644,
                0.7803,
                0.7548,
                0.7973,
            ],
            "Parameters": [
                288740,
                1962308,
                14880964,
                24124292,
                9165026,
                8677572,
                8677572,
                8678724,
            ],
            "Training_Time_Minutes": [1.25, 2.07, 19.75, 12.95, 4.66, 8.16, 8.06, 8.11],
        }
        return pd.DataFrame(demo_data)


def create_accuracy_comparison(models_df):
    """Create accuracy comparison chart"""
    # Format accuracy as percentage for display
    formatted_text = models_df["Test_Accuracy"].apply(lambda x: f"{x:.1%}")

    fig = px.bar(
        models_df,
        x="Model",
        y="Test_Accuracy",
        title="Model Accuracy Comparison",
        color="Test_Accuracy",
        color_continuous_scale="Viridis",
        text=formatted_text,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
        yaxis=dict(tickformat=".1%"),  # Format y-axis as percentage
    )
    fig.update_traces(textposition="outside")
    return fig


def create_f1_comparison(models_df):
    """Create F1-score comparison chart"""
    # Format F1-score as percentage for display
    formatted_text = models_df["Test_Macro_F1"].apply(lambda x: f"{x:.1%}")

    fig = px.bar(
        models_df,
        x="Model",
        y="Test_Macro_F1",
        title="Model F1-Score Comparison",
        color="Test_Macro_F1",
        color_continuous_scale="Plasma",
        text=formatted_text,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
        yaxis=dict(tickformat=".1%"),  # Format y-axis as percentage
    )
    fig.update_traces(textposition="outside")
    return fig


def create_efficiency_plot(models_df):
    """Create efficiency comparison (accuracy vs training time)"""
    fig = px.scatter(
        models_df,
        x="Training_Time_Minutes",
        y="Test_Accuracy",
        size="Parameters",
        color="Model",
        title="Model Efficiency: Accuracy vs Training Time",
        hover_data=["Test_Macro_F1", "Parameters"],
    )

    # Update hover template to show formatted numbers
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>"
        + "Accuracy: %{y:.1%}<br>"
        + "Training Time: %{x:.1f} min<br>"
        + "Parameters: %{customdata[1]:,}<br>"
        + "F1-Score: %{customdata[0]:.3f}<br>"
        + "<extra></extra>"
    )

    fig.update_layout(height=500)
    return fig


def create_parameter_comparison(models_df):
    """Create parameter count comparison"""
    # Format numbers with commas for display
    formatted_text = models_df["Parameters"].apply(lambda x: f"{x:,}")

    fig = px.bar(
        models_df,
        x="Model",
        y="Parameters",
        title="Model Complexity (Parameter Count)",
        color="Parameters",
        color_continuous_scale="Blues",
        text=formatted_text,
    )

    # Calculate max value and add 20% margin for better display
    max_params = models_df["Parameters"].max()
    y_max = max_params * 1.2

    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,  # Increased height
        showlegend=False,
        yaxis=dict(
            range=[0, y_max],  # Set y-axis range with margin
            tickformat=",.0f",  # Format y-axis ticks with commas
        ),
    )
    fig.update_traces(textposition="outside")
    return fig


def main():
    st.title("Facial Emotion Recognition - Project Results")
    st.markdown(
        "**Comprehensive analysis of deep learning models for emotion recognition**"
    )

    # Load data
    models_df = load_results()

    # Sidebar
    st.sidebar.header("Project Overview")
    # Get best model info
    best_model = models_df.loc[models_df["Test_Accuracy"].idxmax()]

    st.sidebar.markdown(
        f"""
    **Dataset**: 4 emotion classes
    - Happy, Sad, Neutral, Surprise

    **Total Images**: ~20,000
    - Training: ~15,000
    - Validation: ~5,000
    - Test: 128 (balanced)

    **Best Model**: {best_model['Model']}
    - **Accuracy**: {best_model['Test_Accuracy']:.1%}
    - **F1-Score**: {best_model['Test_Macro_F1']:.1%}
    - **Parameters**: {best_model['Parameters']:,}
    """
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Performance Summary")

        # Top 3 models
        top_models = models_df.nlargest(3, "Test_Accuracy")
        st.markdown("### Top 3 Performing Models:")

        for i, (_, model) in enumerate(top_models.iterrows(), 1):
            st.markdown(
                f"""
            **{i}. {model['Model']}**
            - Accuracy: {model['Test_Accuracy']:.1%}
            - F1-Score: {model['Test_Macro_F1']:.1%}
            - Parameters: {model['Parameters']:,}
            """
            )

    with col2:
        st.subheader("Key Insights")
        st.markdown(
            """
        • **Custom CNNs** outperform transfer learning

        • **RGB models** show better performance than grayscale

        • **Complex architectures** achieve highest accuracy

        • **Training time** increases with model complexity

        • **Best balance**: Complex RGB CNN (79.69% accuracy)
        """
        )

    # Charts
    st.subheader("Performance Analysis")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Accuracy", "F1-Score", "Efficiency", "Complexity"]
    )

    with tab1:
        st.plotly_chart(
            create_accuracy_comparison(models_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with tab2:
        st.plotly_chart(
            create_f1_comparison(models_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with tab3:
        st.plotly_chart(
            create_efficiency_plot(models_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with tab4:
        st.plotly_chart(
            create_parameter_comparison(models_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Detailed results table
    st.subheader("Detailed Results")

    # Format the dataframe for display
    display_df = models_df.copy()
    display_df["Test_Accuracy"] = display_df["Test_Accuracy"].apply(
        lambda x: f"{x:.1%}"
    )
    display_df["Test_Macro_F1"] = display_df["Test_Macro_F1"].apply(
        lambda x: f"{x:.1%}"
    )
    display_df["Parameters"] = display_df["Parameters"].apply(lambda x: f"{x:,}")
    display_df["Training_Time_Minutes"] = display_df["Training_Time_Minutes"].apply(
        lambda x: f"{x:.1f} min"
    )

    st.dataframe(
        display_df[
            [
                "Model",
                "Test_Accuracy",
                "Test_Macro_F1",
                "Parameters",
                "Training_Time_Minutes",
            ]
        ],
        use_container_width=True,
    )

    # Technical details
    st.subheader("Technical Implementation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Architecture Types:**
        - Custom CNN (3-5 blocks)
        - Transfer Learning (VGG16, ResNet50V2, EfficientNetV2B2)
        - Regularization (L2, Dropout, BatchNorm)

        **Training Features:**
        - Data augmentation
        - Class weight balancing
        - Early stopping
        - Learning rate scheduling
        """
        )

    with col2:
        st.markdown(
            """
        **Evaluation Metrics:**
        - Accuracy
        - Macro F1-Score
        - Weighted F1-Score
        - Confusion Matrix

        **Environment:**
        - TensorFlow/Keras
        - GPU acceleration (Docker)
        - Virtual environment isolation
        - Comprehensive logging
        """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
    **Project Status**: Complete
    **Best Model**: {best_model['Model']} ({best_model['Test_Accuracy']:.1%} accuracy)
    **Total Models Trained**: {len(models_df)}
    **Training Environment**: Docker with GPU support
    **Documentation**: Comprehensive analysis and results available
    """
    )


if __name__ == "__main__":
    main()
