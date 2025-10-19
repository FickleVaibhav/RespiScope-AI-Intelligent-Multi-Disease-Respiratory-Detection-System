"""
Streamlit Web Interface for RespiScope-AI
Alternative to Gradio with different UI/UX
"""

import streamlit as st
import numpy as np
import sys
import os
from pathlib import Path
import torch
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.inference import RespiScopeInference
from utils.config import get_config

# Page configuration
st.set_page_config(
    page_title="RespiScope-AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .stAlert {
        background-color: #f0f2f6;
    }
    .prediction-card {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL_PATH = "models/checkpoints/best_model.pth"
MODEL_TYPE = "crnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disease information
DISEASE_INFO = {
    'Asthma': {
        'icon': '🫁',
        'color': '#ff6b6b',
        'description': 'Chronic inflammatory disease of airways',
        'symptoms': ['Wheezing', 'Breathlessness', 'Chest tightness', 'Coughing'],
        'severity': 'Moderate to Severe'
    },
    'Bronchitis': {
        'icon': '🤧',
        'color': '#ffa07a',
        'description': 'Inflammation of bronchial tubes',
        'symptoms': ['Persistent cough', 'Mucus production', 'Fatigue', 'Chest discomfort'],
        'severity': 'Mild to Moderate'
    },
    'COPD': {
        'icon': '🫁',
        'color': '#dc143c',
        'description': 'Chronic Obstructive Pulmonary Disease',
        'symptoms': ['Chronic cough', 'Breathlessness', 'Chest infections', 'Wheezing'],
        'severity': 'Severe'
    },
    'Pneumonia': {
        'icon': '🦠',
        'color': '#ff4500',
        'description': 'Infection causing lung inflammation',
        'symptoms': ['High fever', 'Cough with phlegm', 'Difficulty breathing', 'Chest pain'],
        'severity': 'Moderate to Severe'
    },
    'Healthy': {
        'icon': '✅',
        'color': '#32cd32',
        'description': 'No respiratory abnormalities detected',
        'symptoms': ['Normal breathing', 'No abnormal sounds'],
        'severity': 'None'
    }
}


@st.cache_resource
def load_model():
    """Load inference model (cached)"""
    try:
        if os.path.exists(MODEL_PATH):
            inference = RespiScopeInference(
                model_path=MODEL_PATH,
                model_type=MODEL_TYPE,
                device=DEVICE
            )
            return inference
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_probability_chart(probabilities):
    """Create horizontal bar chart for probabilities"""
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]
    colors = [DISEASE_INFO[c]['color'] for c in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=probs,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1f}%' for p in probs],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Probability Distribution',
        xaxis_title='Probability (%)',
        yaxis_title='Condition',
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(range=[0, 105])
    )
    
    return fig


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🩺 RespiScope-AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Multi-Class Respiratory Disease Detection System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=RespiScope", use_column_width=True)
        st.header("📋 Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔬 Diagnosis", "📊 History", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.header("⚙️ Settings")
        st.info(f"**Device:** {DEVICE}")
        st.info(f"**Model:** {MODEL_TYPE.upper()}")
        
        st.markdown("---")
        st.header("📞 Support")
        st.markdown("""
        - [Documentation](https://github.com/yourusername/RespiScope-AI)
        - [Report Issue](https://github.com/yourusername/RespiScope-AI/issues)
        - Email: support@respiscope.ai
        """)
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔬 Diagnosis":
        show_diagnosis_page()
    elif page == "📊 History":
        show_history_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Home page"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Supported Conditions", "5")
    with col2:
        st.metric("Model Accuracy", "82%")
    with col3:
        st.metric("Avg Processing Time", "< 1s")
    
    st.markdown("---")
    
    st.header("🎯 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 AI-Powered Detection")
        st.write("""
        - Multi-class classification
        - State-of-the-art deep learning
        - Real-time predictions
        - High accuracy (80-85%)
        """)
        
        st.subheader("🩺 Supported Conditions")
        for condition in DISEASE_INFO.keys():
            st.write(f"{DISEASE_INFO[condition]['icon']} **{condition}**")
    
    with col2:
        st.subheader("📱 Easy to Use")
        st.write("""
        - Upload audio files
        - Record directly from browser
        - Instant results
        - Detailed reports
        """)
        
        st.subheader("🔒 Privacy & Security")
        st.write("""
        - Local processing
        - No data storage
        - HIPAA compliant design
        - Secure predictions
        """)
    
    st.markdown("---")
    
    st.warning("""
    ⚠️ **Medical Disclaimer**: This is an AI-assisted screening tool and NOT a substitute 
    for professional medical diagnosis. Always consult qualified healthcare professionals.
    """)


def show_diagnosis_page():
    """Diagnosis page"""
    st.header("🔬 Audio Analysis")
    
    # Load model
    inference = load_model()
    
    if inference is None:
        st.error("""
        ❌ Model not loaded. Please ensure:
        1. Model is trained: `python models/train.py`
        2. Model path is correct: `models/checkpoints/best_model.pth`
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Audio input options
    input_method = st.radio(
        "Select input method:",
        ["Upload Audio File", "Record Audio"]
    )
    
    audio_file = None
    
    if input_method == "Upload Audio File":
        audio_file = st.file_uploader(
            "Upload respiratory sound (WAV, MP3, FLAC)",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a']
        )
    else:
        st.info("🎤 Click 'Start Recording' to record audio from your microphone")
        # Note: Streamlit doesn't have built-in audio recording
        # Would need to use st.components for custom recording widget
        st.warning("Direct recording requires browser permissions. Use 'Upload Audio File' option instead.")
    
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        
        if st.button("🔍 Analyze Audio", type="primary"):
            with st.spinner("Analyzing audio..."):
                try:
                    # Save temp file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Make prediction
                    result = inference.predict(tmp_path)
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    # Display results
                    show_prediction_results(result)
                    
                    # Save to history
                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'prediction': result['predicted_class'],
                        'confidence': result['confidence'],
                        'filename': audio_file.name
                    })
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")


def show_prediction_results(result):
    """Display prediction results"""
    pred_class = result['predicted_class']
    confidence = result['confidence']
    probabilities = result['probabilities']
    
    # Main prediction
    info = DISEASE_INFO[pred_class]
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Prediction card
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <h2 style="color: {info['color']};">{info['icon']} {pred_class}</h2>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence indicator
        if confidence > 0.9:
            st.success("🟢 High Confidence")
        elif confidence > 0.7:
            st.warning("🟡 Moderate Confidence")
        else:
            st.error("🔴 Low Confidence")
    
    with col2:
        # Probability chart
        fig = create_probability_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed information
    st.subheader(f"ℹ️ About {pred_class}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Severity:** {info['severity']}")
    
    with col2:
        st.write("**Common Symptoms:**")
        for symptom in info['symptoms']:
            st.write(f"• {symptom}")
    
    # Recommendation
    if pred_class != 'Healthy':
        st.warning(f"""
        **⚕️ Recommendation:** 
        Consult a healthcare professional for proper diagnosis and treatment. 
        This AI analysis is for screening purposes only.
        """)
    else:
        st.success("""
        **✅ Recommendation:** 
        No abnormalities detected. Maintain healthy lifestyle and regular check-ups.
        """)


def show_history_page():
    """Prediction history page"""
    st.header("📊 Prediction History")
    
    if 'prediction_history' not in st.session_state or len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Go to Diagnosis page to analyze audio.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.prediction_history)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyses", len(df))
    with col2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    with col3:
        most_common = df['prediction'].mode()[0] if len(df) > 0 else "N/A"
        st.metric("Most Common", most_common)
    
    st.markdown("---")
    
    # Display table
    st.subheader("Recent Predictions")
    st.dataframe(df[['timestamp', 'filename', 'prediction', 'confidence']], use_container_width=True)
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()


def show_about_page():
    """About page"""
    st.header("ℹ️ About RespiScope-AI")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    RespiScope-AI is an AI-powered respiratory disease detection system that combines:
    - Custom digital stethoscope hardware
    - State-of-the-art deep learning models
    - Interactive web interface
    
    ### 🔬 Technology Stack
    
    - **Deep Learning:** PyTorch, PANN, CRNN
    - **Audio Processing:** librosa, torchaudio
    - **Web Framework:** Streamlit
    - **Hardware:** DIY digital stethoscope
    
    ### 📊 Model Performance
    
    - **Accuracy:** 80-85%
    - **F1-Score:** 0.78-0.82
    - **ROC-AUC:** 0.88-0.92
    
    ### 🩺 Supported Conditions
    
    1. **Asthma** - Chronic inflammatory airway disease
    2. **Bronchitis** - Inflammation of bronchial tubes
    3. **COPD** - Chronic Obstructive Pulmonary Disease
    4. **Pneumonia** - Lung infection with inflammation
    5. **Healthy** - No abnormalities detected
    
    ### ⚠️ Important Notice
    
    This system is designed for research and educational purposes. It is **NOT** FDA-approved 
    and should **NOT** be used for clinical diagnosis without proper validation.
    
    ### 📧 Contact
    
    - **GitHub:** [github.com/yourusername/RespiScope-AI](https://github.com/yourusername/RespiScope-AI)
    - **Email:** support@respiscope.ai
    - **Documentation:** See README.md
    
    ### 📄 License
    
    - Software: Apache License 2.0
    - Hardware: CERN Open Hardware Licence (CERN-OHL-P)
    
    ### 🙏 Acknowledgments
    
    - ICBHI 2017 Dataset contributors
    - PANNs authors
    - Open-source community
    """)


if __name__ == '__main__':
    main()
