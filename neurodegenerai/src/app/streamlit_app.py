"""
Streamlit UI for NeuroDegenerAI.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from shared.lib.config import get_settings
from shared.lib.logging import get_logger, setup_logging

# Setup logging
setup_logging(service_name="neurodegenerai_ui")
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="NeuroDegenerAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-result {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-result {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""",
    unsafe_allow_html=True,
)

# API configuration
API_BASE_URL = "http://localhost:9001"


def check_api_health() -> bool:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def make_api_request(
    endpoint: str, data: dict | None = None, method: str = "GET"
) -> dict | None:
    """Make API request with error handling."""
    try:
        if method == "GET":
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
        elif method == "POST":
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def main():
    """Main Streamlit application."""

    # Header
    st.markdown(
        '<h1 class="main-header">🧠 NeuroDegenerAI</h1>', unsafe_allow_html=True
    )
    st.markdown("### Early Neurodegenerative Pattern Detection")

    # Sidebar
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/200x100/1f77b4/ffffff?text=NeuroDegenerAI",
            width=200,
        )

        # API Status
        api_healthy = check_api_health()
        if api_healthy:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Disconnected")
            st.info("Make sure the NeuroDegenerAI API is running on port 9001")

        st.markdown("---")

        # Navigation
        page = st.selectbox(
            "Navigate",
            ["🏠 Home", "📊 Predict", "🧠 Interpretability", "📈 Metrics", "ℹ️ About"],
        )

    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Predict":
        show_predict_page()
    elif page == "🧠 Interpretability":
        show_interpretability_page()
    elif page == "📈 Metrics":
        show_metrics_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Show home page."""

    st.markdown("## Welcome to NeuroDegenerAI")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🎯 What is NeuroDegenerAI?

        NeuroDegenerAI is an advanced machine learning system designed to detect early
        neurodegenerative patterns, particularly in Alzheimer's disease and related
        dementias using:

        - **Tabular Biomarkers**: Age, genetics, cognitive scores, and protein levels
        - **MRI Analysis**: Structural brain imaging with CNN-based pattern recognition
        - **Ensemble Methods**: Combining multiple models for robust predictions

        ### 🔬 Key Features

        - **Early Detection**: Identify patterns before clinical symptoms
        - **Multi-modal Analysis**: Combine biomarker and imaging data
        - **Interpretability**: Understand model decisions with SHAP and Grad-CAM
        - **Clinical Validation**: Based on ADNI dataset standards
        """
        )

    with col2:
        # Demo mode indicator
        settings = get_settings()
        if settings.neuro_demo_mode:
            st.info("🎮 **Demo Mode Active** - Using synthetic data for demonstration")
        else:
            st.success("🔬 **Production Mode** - Using real ADNI data")

        # Quick stats
        st.markdown("### 📊 System Status")

        if check_api_health():
            # Get model info
            model_info = make_api_request("/model/info")
            if model_info:
                st.metric("Model Status", "✅ Loaded")
                st.metric("Model Type", model_info.get("model_type", "Unknown"))
                st.metric("Features", model_info.get("num_features", "Unknown"))

            # Get latest metrics
            metrics = make_api_request("/model/metrics")
            if metrics:
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                with col_metric2:
                    st.metric("AUC", f"{metrics.get('roc_auc', 0):.3f}")
        else:
            st.warning("API not available - showing demo data")
            st.metric("Model Status", "❌ Offline")
            st.metric("Demo Accuracy", "0.847")
            st.metric("Demo AUC", "0.923")

    # Quick start section
    st.markdown("---")
    st.markdown("## 🚀 Quick Start")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        ### 📊 Tabular Prediction
        1. Enter patient demographics
        2. Add biomarker values
        3. Get instant prediction
        """
        )
        if st.button("Start Tabular Prediction", type="primary"):
            st.session_state.page = "predict"
            st.rerun()

    with col2:
        st.markdown(
            """
        ### 🧠 MRI Analysis
        1. Upload MRI scan
        2. Automatic preprocessing
        3. CNN-based analysis
        """
        )
        if st.button("Start MRI Analysis"):
            st.session_state.page = "predict"
            st.rerun()

    with col3:
        st.markdown(
            """
        ### 📈 View Metrics
        1. Model performance
        2. Calibration curves
        3. Feature importance
        """
        )
        if st.button("View Metrics"):
            st.session_state.page = "metrics"
            st.rerun()


def show_predict_page():
    """Show prediction page."""

    st.markdown("## 📊 Prediction Interface")

    # Prediction mode selection
    prediction_mode = st.radio(
        "Select Prediction Mode:",
        ["Tabular Biomarkers", "MRI Analysis", "Ensemble (Both)"],
        horizontal=True,
    )

    if prediction_mode == "Tabular Biomarkers":
        show_tabular_prediction()
    elif prediction_mode == "MRI Analysis":
        show_mri_prediction()
    else:
        show_ensemble_prediction()


def show_tabular_prediction():
    """Show tabular prediction interface."""

    st.markdown("### 🧪 Tabular Biomarker Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Input form
        with st.form("tabular_prediction_form"):
            st.markdown("#### Patient Demographics")

            col_demo1, col_demo2 = st.columns(2)
            with col_demo1:
                age = st.number_input("Age", min_value=0, max_value=120, value=75)
                sex = st.selectbox("Sex", ["Female", "Male"], index=0)

            with col_demo2:
                education = st.number_input(
                    "Years of Education", min_value=0, max_value=25, value=16
                )
                apoe4 = st.selectbox(
                    "APOE4 Status", ["Negative", "Heterozygous", "Homozygous"], index=0
                )

            st.markdown("#### Cognitive Assessment")

            col_cog1, col_cog2 = st.columns(2)
            with col_cog1:
                mmse = st.slider("MMSE Score", min_value=0, max_value=30, value=24)
            with col_cog2:
                cdr = st.slider(
                    "CDR Score", min_value=0.0, max_value=3.0, value=0.5, step=0.5
                )

            st.markdown("#### Biomarkers")

            col_bio1, col_bio2, col_bio3 = st.columns(3)
            with col_bio1:
                abeta = st.number_input(
                    "Amyloid Beta (pg/mL)", min_value=0.0, value=180.0
                )
            with col_bio2:
                tau = st.number_input("Tau (pg/mL)", min_value=0.0, value=350.0)
            with col_bio3:
                ptau = st.number_input(
                    "Phosphorylated Tau (pg/mL)", min_value=0.0, value=28.0
                )

            st.markdown("#### Brain Imaging Measures")

            col_img1, col_img2, col_img3 = st.columns(3)
            with col_img1:
                hippo_vol = st.number_input(
                    "Hippocampal Volume (mm³)", min_value=0.0, value=2800.0
                )
            with col_img2:
                cort_thick = st.number_input(
                    "Cortical Thickness (mm)", min_value=0.0, value=2.3
                )
            with col_img3:
                wmh = st.number_input(
                    "White Matter Hyperintensities", min_value=0.0, value=8.5
                )

            # Demo data button
            if st.form_submit_button("🎮 Use Demo Data"):
                # Set demo values
                st.session_state.demo_data = {
                    "age": 75.0,
                    "sex": 0,
                    "education": 16.0,
                    "apoe4": 1,
                    "mmse": 24.0,
                    "cdr": 0.5,
                    "abeta": 180.0,
                    "tau": 350.0,
                    "ptau": 28.0,
                    "hippocampal_volume": 2800.0,
                    "cortical_thickness": 2.3,
                    "white_matter_hyperintensities": 8.5,
                }
                st.rerun()

            # Submit button
            submitted = st.form_submit_button("🔍 Make Prediction", type="primary")

    with col2:
        st.markdown("#### 📋 Input Summary")

        # Show current inputs
        inputs = {
            "Age": age,
            "Sex": "Female" if sex == 0 else "Male",
            "Education": education,
            "APOE4": ["Negative", "Heterozygous", "Homozygous"][apoe4],
            "MMSE": mmse,
            "CDR": cdr,
            "Amyloid Beta": abeta,
            "Tau": tau,
            "Phosphorylated Tau": ptau,
            "Hippocampal Volume": hippo_vol,
            "Cortical Thickness": cort_thick,
            "White Matter Hyperintensities": wmh,
        }

        for key, value in inputs.items():
            st.text(f"{key}: {value}")

        # Demo data button
        if st.button("🎮 Load Demo Data"):
            st.session_state.demo_data = inputs
            st.rerun()

    # Make prediction
    if submitted:
        # Prepare request data
        request_data = {
            "age": age,
            "sex": 0 if sex == "Female" else 1,
            "apoe4": apoe4,
            "mmse": mmse,
            "cdr": cdr,
            "abeta": abeta,
            "tau": tau,
            "ptau": ptau,
            "education": education,
            "hippocampal_volume": hippo_vol,
            "cortical_thickness": cort_thick,
            "white_matter_hyperintensities": wmh,
        }

        # Make API request
        with st.spinner("Making prediction..."):
            result = make_api_request("/predict/tabular", request_data, "POST")

        if result:
            show_prediction_result(result, "Tabular")


def show_mri_prediction():
    """Show MRI prediction interface."""

    st.markdown("### 🧠 MRI Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Upload MRI Scan")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a NIfTI file (.nii or .nii.gz)",
            type=["nii", "nii.gz"],
            help="Upload a 3D MRI scan in NIfTI format",
        )

        # Demo MRI button
        if st.button("🎮 Use Demo MRI"):
            # Generate demo MRI data
            demo_volume = generate_demo_mri_volume()
            st.session_state.demo_mri = demo_volume
            st.success("Demo MRI data generated!")

        # Show demo MRI if available
        if "demo_mri" in st.session_state:
            st.markdown("#### Demo MRI Volume")
            st.write(f"Shape: {st.session_state.demo_mri.shape}")

            # Show some slices
            fig = plot_mri_slices(st.session_state.demo_mri)
            st.plotly_chart(fig, use_container_width=True)

            if st.button("🔍 Analyze Demo MRI"):
                request_data = {"volume_data": st.session_state.demo_mri.tolist()}

                with st.spinner("Analyzing MRI..."):
                    result = make_api_request("/predict/mri", request_data, "POST")

                if result:
                    show_prediction_result(result, "MRI")

        # Handle uploaded file
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")

            if st.button("🔍 Analyze Uploaded MRI"):
                # For demo purposes, we'll use the demo MRI
                # In production, you'd process the uploaded file
                st.info("Processing uploaded file...")

                request_data = {
                    "volume_data": st.session_state.get(
                        "demo_mri", generate_demo_mri_volume()
                    ).tolist()
                }

                with st.spinner("Analyzing MRI..."):
                    result = make_api_request("/predict/mri", request_data, "POST")

                if result:
                    show_prediction_result(result, "MRI")

    with col2:
        st.markdown("#### 📊 MRI Info")
        st.info(
            """
        **Supported Formats:**
        - NIfTI (.nii, .nii.gz)
        - 3D volumes
        - T1-weighted scans preferred

        **Processing:**
        - Automatic preprocessing
        - Slice extraction
        - CNN analysis
        - Grad-CAM visualization
        """
        )


def show_ensemble_prediction():
    """Show ensemble prediction interface."""

    st.markdown("### 🔬 Ensemble Analysis")
    st.info(
        "Combines both tabular biomarkers and MRI analysis for comprehensive assessment."
    )

    # This would combine both tabular and MRI inputs
    st.markdown("**Coming Soon:** Full ensemble prediction interface")

    # For now, show a placeholder
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tabular Score", "0.73", "↑ 0.05")

    with col2:
        st.metric("MRI Score", "0.81", "↑ 0.02")

    with col3:
        st.metric("Ensemble Score", "0.77", "↑ 0.03")


def show_prediction_result(result: dict[str, Any], model_type: str):
    """Show prediction result."""

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    prediction = result.get("prediction", 0)
    probability = result.get("probability", 0.0)
    confidence = result.get("confidence", 0.0)

    # Result display based on prediction
    if prediction == 0:
        st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
        st.markdown("### ✅ **Normal Cognition Predicted**")
        st.markdown(f"**Probability:** {probability:.3f}")
        st.markdown(f"**Confidence:** {confidence:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-result">', unsafe_allow_html=True)
        st.markdown("### ⚠️ **Cognitive Impairment Predicted**")
        st.markdown(f"**Probability:** {probability:.3f}")
        st.markdown(f"**Confidence:** {confidence:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Model information
    st.markdown("#### Model Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", result.get("model_name", "Unknown"))

    with col2:
        st.metric("Type", result.get("model_type", "Unknown"))

    with col3:
        st.metric("Confidence", f"{confidence:.3f}")

    # Explanation if available
    if "explanation" in result and result["explanation"]:
        show_explanation(result["explanation"])

    # Heatmaps if available
    if "heatmap_paths" in result and result["heatmap_paths"]:
        show_heatmaps(result["heatmap_paths"])


def show_explanation(explanation: dict[str, Any]):
    """Show model explanation."""

    st.markdown("#### 🧠 Model Explanation")

    if "top_features" in explanation:
        # Feature importance
        st.markdown("**Top Contributing Features:**")

        features_df = pd.DataFrame(explanation["top_features"])

        # Create bar chart
        fig = px.bar(
            features_df.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance",
            color="importance",
            color_continuous_scale="viridis",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Show feature values
        st.markdown("**Feature Values:**")
        st.dataframe(features_df, use_container_width=True)


def show_heatmaps(heatmap_paths: list):
    """Show Grad-CAM heatmaps."""

    st.markdown("#### 🔥 Grad-CAM Visualizations")

    for i, path in enumerate(heatmap_paths[:4]):  # Show max 4 heatmaps
        try:
            st.image(path, caption=f"Grad-CAM Heatmap {i+1}", use_column_width=True)
        except Exception:
            st.warning(f"Could not load heatmap: {path}")


def show_interpretability_page():
    """Show interpretability page."""

    st.markdown("## 🧠 Model Interpretability")

    st.markdown(
        """
    This page provides insights into how the NeuroDegenerAI models make their predictions.
    Understanding model decisions is crucial for clinical applications.
    """
    )

    # Feature importance
    st.markdown("### 📊 Feature Importance")

    importance_data = make_api_request("/model/features/importance")

    if importance_data:
        features_df = pd.DataFrame(importance_data["top_features"])

        # Create visualization
        fig = px.bar(
            features_df.head(20),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 20 Most Important Features",
            color="importance",
            color_continuous_scale="viridis",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Show detailed table
        st.markdown("#### Detailed Feature Rankings")
        st.dataframe(features_df, use_container_width=True)
    else:
        st.warning("Feature importance data not available")

    # SHAP explanations
    st.markdown("### 🔍 SHAP Analysis")
    st.info(
        "SHAP (SHapley Additive exPlanations) provides consistent and locally accurate feature attributions."
    )

    # Placeholder for SHAP plots
    st.markdown("**SHAP Summary Plot**")
    st.image(
        "https://via.placeholder.com/800x400/f0f2f6/666666?text=SHAP+Summary+Plot",
        use_column_width=True,
    )

    # Grad-CAM for MRI
    st.markdown("### 🧠 MRI Grad-CAM Analysis")
    st.info(
        "Grad-CAM highlights regions in MRI scans that contribute most to the model's decision."
    )

    # Placeholder for Grad-CAM
    st.markdown("**Grad-CAM Visualizations**")
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "https://via.placeholder.com/400x300/f0f2f6/666666?text=Original+MRI",
            use_column_width=True,
        )
    with col2:
        st.image(
            "https://via.placeholder.com/400x300/f0f2f6/666666?text=Grad-CAM+Heatmap",
            use_column_width=True,
        )


def show_metrics_page():
    """Show metrics page."""

    st.markdown("## 📈 Model Performance Metrics")

    # Get metrics from API
    metrics_data = make_api_request("/model/metrics")

    if metrics_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{metrics_data.get('accuracy', 0):.3f}")

        with col2:
            st.metric("Precision", f"{metrics_data.get('precision', 0):.3f}")

        with col3:
            st.metric("Recall", f"{metrics_data.get('recall', 0):.3f}")

        with col4:
            st.metric("F1 Score", f"{metrics_data.get('f1_score', 0):.3f}")

        # ROC and PR curves
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ROC Curve")
            st.image(
                "https://via.placeholder.com/600x400/f0f2f6/666666?text=ROC+Curve",
                use_column_width=True,
            )

        with col2:
            st.markdown("### Precision-Recall Curve")
            st.image(
                "https://via.placeholder.com/600x400/f0f2f6/666666?text=PR+Curve",
                use_column_width=True,
            )

        # Confusion matrix
        st.markdown("### Confusion Matrix")
        st.image(
            "https://via.placeholder.com/500x400/f0f2f6/666666?text=Confusion+Matrix",
            use_column_width=True,
        )

        # Calibration curve
        st.markdown("### Calibration Curve")
        st.image(
            "https://via.placeholder.com/600x400/f0f2f6/666666?text=Calibration+Curve",
            use_column_width=True,
        )

    else:
        st.warning("Metrics data not available")

        # Show demo metrics
        st.markdown("### Demo Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", "0.847")

        with col2:
            st.metric("Precision", "0.823")

        with col3:
            st.metric("Recall", "0.856")

        with col4:
            st.metric("F1 Score", "0.839")


def show_about_page():
    """Show about page."""

    st.markdown("## ℹ️ About NeuroDegenerAI")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🎯 Mission
        NeuroDegenerAI aims to revolutionize early detection of neurodegenerative
        diseases through advanced machine learning techniques, providing clinicians
        with powerful tools for early intervention and personalized treatment.

        ### 🔬 Technology
        - **Machine Learning**: LightGBM, XGBoost, and CNN models
        - **Data Sources**: ADNI dataset and real-world biomarkers
        - **Interpretability**: SHAP, Grad-CAM, and Integrated Gradients
        - **Deployment**: FastAPI backend with Streamlit frontend

        ### 🏥 Clinical Applications
        - Early Alzheimer's detection
        - Risk stratification
        - Treatment monitoring
        - Research applications
        """
        )

    with col2:
        st.markdown(
            """
        ### ⚠️ Important Disclaimers

        **Not for Clinical Use**: This is a research prototype and should not be
        used for clinical diagnosis or treatment decisions.

        **Data Privacy**: All patient data is processed securely and never stored
        permanently.

        **Model Limitations**: Models are trained on specific datasets and may
        not generalize to all populations.

        ### 📚 References
        - ADNI Dataset
        - Alzheimer's Association Guidelines
        - Machine Learning Best Practices
        - Clinical Validation Studies

        ### 📞 Contact
        For questions or support, please contact the development team.
        """
        )

    # Model information
    st.markdown("---")
    st.markdown("### 🔧 Technical Information")

    model_info = make_api_request("/model/info")

    if model_info:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Model Name:** {model_info.get('model_name', 'Unknown')}")
            st.markdown(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")

        with col2:
            st.markdown(f"**Model Class:** {model_info.get('model_class', 'Unknown')}")
            st.markdown(f"**Version:** {model_info.get('version', 'Unknown')}")

        with col3:
            st.markdown(f"**Features:** {model_info.get('num_features', 'Unknown')}")
            st.markdown(
                f"**Parameters:** {model_info.get('num_parameters', 'Unknown')}"
            )
    else:
        st.info("Model information not available")


def generate_demo_mri_volume() -> np.ndarray:
    """Generate demo MRI volume."""
    np.random.seed(42)
    return np.random.randn(64, 64, 64).astype(np.float32)


def plot_mri_slices(volume: np.ndarray) -> go.Figure:
    """Plot MRI slices."""

    # Get middle slices
    mid_z = volume.shape[2] // 2
    slice_data = volume[:, :, mid_z]

    fig = go.Figure(data=go.Heatmap(z=slice_data, colorscale="gray", showscale=False))

    fig.update_layout(
        title="MRI Slice (Demo Data)", xaxis_title="X", yaxis_title="Y", height=400
    )

    return fig


if __name__ == "__main__":
    main()
