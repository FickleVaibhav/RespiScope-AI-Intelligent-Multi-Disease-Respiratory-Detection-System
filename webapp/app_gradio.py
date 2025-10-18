"""
Gradio Web Interface for RespiScope-AI
Interactive respiratory disease detection system
"""

import gradio as gr
import numpy as np
import sys
import os
from pathlib import Path
import torch
import json
import plotly.graph_objects as go
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.inference import RespiScopeInference
from utils.config import get_config


# Configuration
MODEL_PATH = "models/checkpoints/best_model.pth"
MODEL_TYPE = "crnn"  # or "transformer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disease descriptions
DISEASE_INFO = {
    'Asthma': {
        'description': 'A chronic inflammatory disease of the airways characterized by wheezing, breathlessness, chest tightness, and coughing.',
        'symptoms': ['Wheezing', 'Shortness of breath', 'Chest tightness', 'Chronic cough'],
        'severity': 'Moderate to Severe',
        'recommendation': 'Consult a pulmonologist for proper diagnosis and treatment plan.'
    },
    'Bronchitis': {
        'description': 'Inflammation of the bronchial tubes, often following a respiratory infection.',
        'symptoms': ['Persistent cough', 'Mucus production', 'Fatigue', 'Chest discomfort'],
        'severity': 'Mild to Moderate',
        'recommendation': 'Rest and consult a doctor if symptoms persist beyond 3 weeks.'
    },
    'COPD': {
        'description': 'Chronic Obstructive Pulmonary Disease - a progressive lung disease causing breathing difficulties.',
        'symptoms': ['Chronic cough', 'Increased breathlessness', 'Frequent chest infections', 'Wheezing'],
        'severity': 'Severe',
        'recommendation': 'Immediate medical attention required. Consult a pulmonologist.'
    },
    'Pneumonia': {
        'description': 'Infection of the lungs causing inflammation of air sacs, which may fill with fluid.',
        'symptoms': ['High fever', 'Cough with phlegm', 'Difficulty breathing', 'Chest pain'],
        'severity': 'Moderate to Severe',
        'recommendation': 'Seek immediate medical care. May require hospitalization.'
    },
    'Healthy': {
        'description': 'No significant respiratory abnormalities detected.',
        'symptoms': ['Normal breathing patterns', 'No wheezing or crackles'],
        'severity': 'None',
        'recommendation': 'Maintain healthy lifestyle and regular check-ups.'
    }
}


class RespiScopeApp:
    """Main application class"""
    
    def __init__(self, model_path: str, model_type: str, device: str):
        """Initialize the application"""
        self.inference = None
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.prediction_history = []
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the inference model"""
        try:
            if os.path.exists(self.model_path):
                self.inference = RespiScopeInference(
                    model_path=self.model_path,
                    model_type=self.model_type,
                    device=self.device
                )
                print(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                print(f"Warning: Model not found at {self.model_path}")
                print("Please train a model first or update MODEL_PATH")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_audio(self, audio_input):
        """
        Make prediction on audio input
        
        Args:
            audio_input: Tuple of (sample_rate, audio_data) from Gradio
            
        Returns:
            Tuple of (prediction_text, probability_plot, info_text)
        """
        if self.inference is None:
            return (
                "❌ Model not loaded. Please check model path and train a model first.",
                None,
                "Model Error"
            )
        
        try:
            # Extract audio data
            if audio_input is None:
                return "⚠️ Please upload or record an audio file.", None, ""
            
            sample_rate, audio_data = audio_input
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Make prediction
            result = self.inference.predict_from_array(
                audio_data,
                sample_rate=sample_rate
            )
            
            # Store in history
            self.prediction_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': result['predicted_class'],
                'confidence': result['confidence']
            })
            
            # Create prediction text
            pred_class = result['predicted_class']
            confidence = result['confidence']
            
            # Confidence emoji
            if confidence > 0.9:
                conf_emoji = "🟢"
            elif confidence > 0.7:
                conf_emoji = "🟡"
            else:
                conf_emoji = "🔴"
            
            prediction_text = f"""
## 🎯 Prediction Results

### Detected Condition: **{pred_class}**
{conf_emoji} **Confidence:** {confidence:.2%}

---
"""
            
            # Create probability plot
            prob_plot = self.create_probability_plot(result['probabilities'])
            
            # Create info text
            info = DISEASE_INFO[pred_class]
            info_text = f"""
## 📋 Condition Information

**Description:** {info['description']}

**Common Symptoms:**
{chr(10).join('• ' + s for s in info['symptoms'])}

**Severity:** {info['severity']}

**Recommendation:** {info['recommendation']}

---

⚠️ **Disclaimer:** This is an AI-assisted screening tool and NOT a substitute for professional medical diagnosis. 
Always consult with qualified healthcare professionals for accurate diagnosis and treatment.
"""
            
            return prediction_text, prob_plot, info_text
            
        except Exception as e:
            return f"❌ Error during prediction: {str(e)}", None, ""
    
    def create_probability_plot(self, probabilities: dict):
        """Create interactive probability bar plot"""
        classes = list(probabilities.keys())
        probs = [probabilities[c] * 100 for c in classes]
        
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        classes = [classes[i] for i in sorted_indices]
        probs = [probs[i] for i in sorted_indices]
        
        # Create color scale
        colors = ['#FF6B6B' if p < 50 else '#4ECDC4' if p < 80 else '#45B7D1' 
                  for p in probs]
        
        fig = go.Figure(data=[
            go.Bar(
                x=probs,
                y=classes,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            xaxis=dict(range=[0, 105])
        )
        
        return fig
    
    def get_history(self):
        """Get prediction history"""
        if not self.prediction_history:
            return "No predictions yet."
        
        history_text = "## 📊 Prediction History\n\n"
        for i, record in enumerate(reversed(self.prediction_history[-10:]), 1):
            history_text += f"{i}. **{record['timestamp']}** - {record['prediction']} ({record['confidence']:.2%})\n"
        
        return history_text
    
    def clear_history(self):
        """Clear prediction history"""
        self.prediction_history = []
        return "History cleared."
    
    def export_results(self, audio_input):
        """Export prediction results as JSON"""
        if audio_input is None:
            return None
        
        try:
            sample_rate, audio_data = audio_input
            
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            audio_data = audio_data.astype(np.float32)
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            result = self.inference.predict_from_array(
                audio_data,
                sample_rate=sample_rate
            )
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'prediction': result['predicted_class'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'model_type': self.model_type,
                'device': self.device
            }
            
            # Save to file
            filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return filename
            
        except Exception as e:
            print(f"Export error: {e}")
            return None


def create_interface():
    """Create Gradio interface"""
    
    # Initialize app
    app = RespiScopeApp(MODEL_PATH, MODEL_TYPE, DEVICE)
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    .prediction-box {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
    }
    """
    
    # Create interface
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🩺 RespiScope-AI: Respiratory Disease Detection System
        
        Upload or record respiratory sounds (cough, breathing) to detect potential conditions.
        
        **Supported Conditions:** Asthma, Bronchitis, COPD, Pneumonia, Healthy
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Audio Input")
                
                # Audio input
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="numpy",
                    label="Upload or Record Audio",
                    format="wav"
                )
                
                with gr.Row():
                    predict_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", size="lg")
                
                gr.Markdown("""
                ### 📝 Instructions
                1. **Record:** Click microphone icon and record cough/breathing sounds (5-10 seconds)
                2. **Upload:** Or upload an audio file (WAV, MP3, FLAC)
                3. **Analyze:** Click the Analyze button to get prediction
                4. **Review:** Check the results and recommendations
                
                **Tips:**
                - Record in a quiet environment
                - Position stethoscope properly on chest
                - Breathe deeply and cough naturally
                - Multiple recordings improve accuracy
                """)
        
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                # Prediction output
                prediction_output = gr.Markdown(
                    label="Prediction",
                    value="Upload or record audio to begin..."
                )
                
                # Probability plot
                probability_plot = gr.Plot(label="Probability Distribution")
                
                # Condition info
                with gr.Accordion("ℹ️ Detailed Information", open=False):
                    info_output = gr.Markdown()
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📥 Export & History")
                
                with gr.Row():
                    export_btn = gr.Button("💾 Export Results (JSON)")
                    history_btn = gr.Button("📊 View History")
                    clear_history_btn = gr.Button("🗑️ Clear History")
                
                export_file = gr.File(label="Downloaded Results")
                history_output = gr.Markdown()
        
        gr.Markdown("""
        ---
        ### ⚠️ Important Disclaimer
        
        This AI system is designed for **screening and educational purposes only**. It is **NOT** a medical device 
        and should **NOT** be used as a substitute for professional medical diagnosis, treatment, or advice.
        
        **Always consult qualified healthcare professionals** for accurate diagnosis and appropriate treatment 
        of respiratory conditions.
        
        ---
        
        ### 🔬 Technical Information
        - **Model:** PANN + CRNN/Transformer
        - **Classes:** 5 (Asthma, Bronchitis, COPD, Pneumonia, Healthy)
        - **Sample Rate:** 16 kHz
        - **Device:** {device}
        
        **Version:** 1.0.0 | **License:** Apache 2.0
        """.format(device=DEVICE))
        
        # Event handlers
        predict_btn.click(
            fn=app.predict_audio,
            inputs=[audio_input],
            outputs=[prediction_output, probability_plot, info_output]
        )
        
        clear_btn.click(
            fn=lambda: (None, "Upload or record audio to begin...", None, ""),
            inputs=[],
            outputs=[audio_input, prediction_output, probability_plot, info_output]
        )
        
        export_btn.click(
            fn=app.export_results,
            inputs=[audio_input],
            outputs=[export_file]
        )
        
        history_btn.click(
            fn=app.get_history,
            inputs=[],
            outputs=[history_output]
        )
        
        clear_history_btn.click(
            fn=app.clear_history,
            inputs=[],
            outputs=[history_output]
        )
    
    return interface


def main():
    """Launch the application"""
    print("=" * 80)
    print("RespiScope-AI Web Interface")
    print("=" * 80)
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Device: {DEVICE}")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 80)
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )


if __name__ == "__main__":
    main()
