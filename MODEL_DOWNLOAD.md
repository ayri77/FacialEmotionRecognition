# Model Files Download

The trained model files are too large for GitHub (>100MB). To run the application locally, you need to download the models separately.

## Required Models

### For Real-time Detection:
- `models/converted_best_hpo_optimized.keras` (12.4M parameters, best performance)
- `models/best_hpo_optimized.keras` (original HPO model)
- `models/demo_model.keras` (demo model for testing)

### For Model Analysis:
- `models/best_complex_cnn_rgb.keras` (79.69% accuracy)
- `models/best_complex_cnn.keras` (78.13% accuracy)
- `models/best_complex_cnn_gs192_strong_reg.keras` (77.34% accuracy)

## How to Get Models

1. **From Docker Environment**: If you have the original Docker setup, copy models from `docker/models/` to `models/`

2. **From Training**: Run the training notebooks in `notebooks/Experiments/` to generate models

3. **Demo Model**: The application will create a demo model automatically if no trained models are found

## Model Information

- **Best Model**: HPO Optimized CNN (converted_best_hpo_optimized.keras)
- **Accuracy**: 79.69% on test set
- **Parameters**: 12.4M
- **Input Size**: 48x48x3 (RGB)
- **Classes**: Happy, Neutral, Sad, Surprise

## Streamlit Cloud

For Streamlit Cloud deployment, the application will use the demo model if trained models are not available. The demo model provides random predictions for demonstration purposes.
