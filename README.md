# RespiScope-AI-Intelligent-Multi-Disease-Respiratory-Detection-System
RespiScope-AI is an advanced AI-powered diagnostic system that detects multiple respiratory diseases — such as asthma, bronchitis, pneumonia, COPD, and healthy lung conditions — using cough and breathing sounds captured through a smart digital stethoscope or built-in microphone.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A complete AI-powered respiratory disease detection system using cough and breathing sounds captured through a custom digital stethoscope. Detects **Asthma, Bronchitis, COPD, Pneumonia, and Healthy** conditions.

## 🎯 Project Overview

RespiScope-AI combines hardware and software to create an affordable, accurate respiratory disease screening tool suitable for:
- Clinical screening and triage
- Remote patient monitoring
- Research and education
- Science fair and final-year projects

**Key Features:**
- 🔬 Multi-class classification (5 respiratory conditions)
- 🎵 PANN-based audio feature extraction
- 🧠 CRNN/Transformer architecture for classification
- 🔧 DIY digital stethoscope (no microcontroller needed)
- 🌐 Interactive web interface (Gradio & Streamlit)
- 📊 Comprehensive evaluation metrics
- 📚 Multiple public dataset support

---

## 📁 Repository Structure

```
RespiScope-AI/
├── hardware/
│   ├── stethoscope_assembly/
│   │   ├── assembly_steps.md
│   │   ├── images/
│   │   │   ├── step1_components.jpg
│   │   │   ├── step2_microphone.jpg
│   │   │   ├── step3_wiring.jpg
│   │   │   └── final_assembly.jpg
│   │   └── bill_of_materials.csv
│   ├── microphone_connection/
│   │   ├── schematic_trrs.png
│   │   ├── schematic_usb.png
│   │   └── wiring_guide.md
│   └── setup_guide.pdf
├── datasets/
│   ├── download_scripts/
│   │   ├── download_icbhi.py
│   │   ├── download_coswara.py
│   │   └── download_all.sh
│   └── preprocessing/
│       ├── audio_preprocessing.py
│       ├── data_augmentation.py
│       └── prepare_splits.py
├── models/
│   ├── pann_embeddings.py
│   ├── crnn_classifier.py
│   ├── transformer_classifier.py
│   ├── train.py
│   ├── inference.py
│   └── checkpoints/
├── webapp/
│   ├── app_gradio.py
│   ├── app_streamlit.py
│   ├── static/
│   └── requirements.txt
├── docs/
│   ├── architecture_diagram.png
│   ├── dataset_description.md
│   ├── training_guide.md
│   └── project_report.pdf
├── utils/
│   ├── audio_utils.py
│   ├── metrics.py
│   ├── config.py
│   └── logger.py
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_comparison.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_inference.py
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🔧 Hardware Assembly

### Components Required (Bill of Materials)

| Component | Specification | Quantity | Est. Cost (USD) |
|-----------|--------------|----------|-----------------|
| Analog Stethoscope | Standard dual-head | 1 | $15-30 |
| Electret Microphone | High-sensitivity (e.g., Adafruit 1713) | 1 | $7 |
| TRRS Audio Jack | 3.5mm 4-pole male | 1 | $2 |
| Heat Shrink Tubing | Various sizes | 1 set | $5 |
| Silicone Sealant | Waterproof | 1 tube | $5 |
| Wires | 22-24 AWG | 1m | $2 |
| USB Audio Adapter | (Optional) for better quality | 1 | $10-20 |

**Total Cost: ~$46-71**

### Assembly Instructions

See detailed step-by-step guide in `hardware/stethoscope_assembly/assembly_steps.md`

**Quick Overview:**
1. **Disassemble stethoscope chest piece** - Remove the diaphragm
2. **Mount microphone** - Secure inside chest piece facing diaphragm
3. **Solder connections** - Wire microphone to TRRS jack
4. **Seal and insulate** - Use heat shrink and silicone sealant
5. **Reassemble** - Put chest piece back together
6. **Test connection** - Verify audio input on PC/smartphone

### Connection Methods

**Option A: TRRS 3.5mm Jack** (Smartphone/PC compatible)
- Simple plug-and-play
- Works with most devices
- See `hardware/microphone_connection/schematic_trrs.png`

**Option B: USB Audio Interface** (Higher quality)
- Better signal-to-noise ratio
- Requires USB audio adapter
- See `hardware/microphone_connection/schematic_usb.png`

---

## 🤖 AI Architecture

### Overview

```
Audio Input (16kHz) 
    ↓
Preprocessing (Normalization, Trimming)
    ↓
PANN Feature Extraction (PANNs CNN14)
    ↓
Embeddings (2048-dimensional)
    ↓
CRNN/Transformer Classifier
    ↓
Multi-class Output (5 classes)
    ↓
[Asthma | Bronchitis | COPD | Pneumonia | Healthy]
```

### Model Components

1. **PANN (Pretrained Audio Neural Networks)**
   - Pretrained on AudioSet (2M+ audio clips)
   - Extracts rich audio embeddings
   - CNN14 architecture (14-layer CNN)

2. **CRNN Classifier**
   - CNN layers for spatial features
   - Bi-directional LSTM for temporal modeling
   - Fully connected layers for classification
   - Dropout and batch normalization for regularization

3. **Transformer Classifier** (Alternative)
   - Multi-head self-attention
   - Positional encoding for temporal information
   - Better for long-range dependencies

### Performance Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Per-class and weighted average
- **Confusion Matrix**: Class-wise performance
- **ROC-AUC**: One-vs-rest for each class
- **Precision/Recall**: Per-class metrics

---

## 📊 Datasets

### Supported Datasets

1. **ICBHI 2017 Respiratory Sound Database**
   - 920 audio recordings
   - 126 subjects
   - Crackles, wheezes, and normal sounds
   - Multiple respiratory conditions

2. **Coswara Dataset**
   - COVID-19 respiratory sounds
   - Cough, breathing, and voice samples
   - Crowdsourced global data

3. **Custom Dataset**
   - Add your own labeled recordings
   - Follow the preprocessing pipeline

### Dataset Statistics

See `docs/dataset_description.md` for detailed information.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/RespiScope-AI.git
cd RespiScope-AI
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Datasets

```bash
cd datasets/download_scripts
python download_icbhi.py
python download_coswara.py
cd ../..
```

### Step 5: Preprocess Data

```bash
cd datasets/preprocessing
python audio_preprocessing.py --input ../../data/raw --output ../../data/processed
python prepare_splits.py --data ../../data/processed --output ../../data/splits
cd ../..
```

---

## 🎓 Training

### Quick Start

```bash
python models/train.py \
    --data_path data/splits \
    --model_type crnn \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --save_dir models/checkpoints
```

### Training Options

```bash
# Use CRNN model
python models/train.py --model_type crnn

# Use Transformer model
python models/train.py --model_type transformer

# Enable data augmentation
python models/train.py --augment

# Resume from checkpoint
python models/train.py --resume models/checkpoints/best_model.pth
```

### Monitor Training

Training logs and metrics are saved to `models/checkpoints/logs/`

Use TensorBoard for visualization:
```bash
tensorboard --logdir models/checkpoints/logs
```

---

## 🔮 Inference

### Command Line Inference

```bash
python models/inference.py \
    --audio_path path/to/audio.wav \
    --model_path models/checkpoints/best_model.pth \
    --model_type crnn
```

### Python API

```python
from models.inference import RespiScopeInference

# Initialize model
model = RespiScopeInference(
    model_path='models/checkpoints/best_model.pth',
    model_type='crnn'
)

# Predict single audio
result = model.predict('path/to/audio.wav')
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probabilities: {result['probabilities']}")
```

---

## 🌐 Web Interface

### Gradio Interface

```bash
cd webapp
python app_gradio.py
```

Access at: `http://localhost:7860`

**Features:**
- 🎙️ Record audio directly from browser
- 📁 Upload audio files (WAV, MP3, FLAC)
- 📊 Real-time predictions with confidence scores
- 📈 Probability distribution visualization
- 💾 Download predictions as JSON

### Streamlit Interface

```bash
cd webapp
streamlit run app_streamlit.py
```

Access at: `http://localhost:8501`

**Features:**
- Similar to Gradio with alternative UI
- Batch processing support
- Historical predictions log

---

## 📈 Evaluation

### Generate Evaluation Report

```bash
python utils/metrics.py \
    --model_path models/checkpoints/best_model.pth \
    --test_data data/splits/test \
    --output_dir results/
```

**Outputs:**
- `confusion_matrix.png`
- `roc_curves.png`
- `classification_report.txt`
- `per_class_metrics.csv`

---

## 📖 Documentation

### Architecture Details

See `docs/architecture_diagram.png` for visual representation.

### Dataset Information

Detailed dataset statistics and preprocessing steps in `docs/dataset_description.md`.

### Training Guide

Complete training pipeline explanation in `docs/training_guide.md`.

### Project Report

Comprehensive academic report in `docs/project_report.pdf`.

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run specific test:

```bash
pytest tests/test_preprocessing.py
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

**Software:** Apache License 2.0 - See [LICENSE](LICENSE) file for details.

**Hardware:** CERN Open Hardware Licence Version 2 - Permissive (CERN-OHL-P)

---

## 🙏 Acknowledgments

- **PANN**: Kong et al. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
- **ICBHI Dataset**: Rocha et al. "An open access database for the evaluation of respiratory sound classification algorithms"
- **Peter Ma's Digital Stethoscope**: Hardware design inspiration
- **AudioSet**: Google's large-scale audio dataset for pretraining

---

## 📧 Contact

**Project Maintainer**: Vaibhav Kumar Mohanty
- Email: mohantykvaibhav.email@example.com
- GitHub: [@FickleVaibhav](https://github.com/FickleVaibhav)

**Issues**: Please report bugs and feature requests on [GitHub Issues](https://github.com/FickleVaibhav/RespiScope-AI/issues)

---

## 🎯 Citation

If you use this project in your research, please cite:

```bibtex
@software{respiscope_ai_2024,
  title={RespiScope-AI: Multi-Class Respiratory Disease Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/RespiScope-AI}
}
```

---

## 🔮 Future Enhancements

- [ ] Mobile app (Android/iOS)
- [ ] Edge deployment (Raspberry Pi, NVIDIA Jetson)
- [ ] Real-time continuous monitoring
- [ ] Multi-language support
- [ ] Cloud-based API
- [ ] Integration with EHR systems
- [ ] Explainable AI (Grad-CAM for audio)

---

## ⚠️ Disclaimer

This system is intended for research and educational purposes only. It is **NOT** a medical device and should **NOT** be used for clinical diagnosis without proper validation and regulatory approval. Always consult qualified healthcare professionals for medical advice.

---

**Version**: 1.0.0  
**Last Updated**: 25 October 2025
**Status**: Active Development
