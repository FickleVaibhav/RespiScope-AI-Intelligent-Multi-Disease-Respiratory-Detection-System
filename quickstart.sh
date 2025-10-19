#!/bin/bash

# RespiScope-AI Quick Start Script
# Automates setup, dataset download, and initial training

set -e  # Exit on error

echo "=================================="
echo "RespiScope-AI Quick Start"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{raw,processed,splits}/{icbhi,coswara}
mkdir -p models/checkpoints
mkdir -p logs
mkdir -p results
mkdir -p hardware/stethoscope_assembly/images
mkdir -p hardware/microphone_connection
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check for CUDA
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA detected${NC}"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo -e "${YELLOW}⚠ CUDA not detected. Will use CPU (slower training)${NC}"
fi
echo ""

# Setup Kaggle API
echo "Setting up Kaggle API..."
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "${YELLOW}⚠ Kaggle API credentials not found${NC}"
    echo "To download ICBHI dataset, please:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Create API token (downloads kaggle.json)"
    echo "3. Place in ~/.kaggle/kaggle.json"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Do you want to continue without downloading datasets? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Kaggle API credentials found${NC}"
    
    # Download ICBHI dataset
    read -p "Download ICBHI dataset? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading ICBHI dataset..."
        python datasets/download_scripts/download_icbhi.py
        echo -e "${GREEN}✓ ICBHI dataset downloaded${NC}"
    fi
fi
echo ""

# Preprocess data
if [ -d "data/raw/icbhi" ] && [ "$(ls -A data/raw/icbhi)" ]; then
    read -p "Preprocess ICBHI dataset? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Preprocessing ICBHI dataset..."
        python datasets/preprocessing/audio_preprocessing.py --dataset icbhi
        echo -e "${GREEN}✓ Preprocessing complete${NC}"
        
        # Prepare splits
        read -p "Prepare train/val/test splits? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Preparing data splits..."
            python datasets/preprocessing/prepare_splits.py
            echo -e "${GREEN}✓ Data splits created${NC}"
        fi
    fi
fi
echo ""

# Generate default config
echo "Generating configuration file..."
python -c "from utils.config import get_config, save_config; save_config(get_config(), 'config.yaml')"
echo -e "${GREEN}✓ Configuration file created: config.yaml${NC}"
echo ""

# Download pretrained PANN weights
echo "Downloading pretrained PANN weights..."
mkdir -p checkpoints
if [ ! -f "checkpoints/Cnn14_mAP=0.431.pth" ]; then
    wget -q --show-progress https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth -O checkpoints/Cnn14_mAP=0.431.pth
    echo -e "${GREEN}✓ PANN weights downloaded${NC}"
else
    echo -e "${YELLOW}PANN weights already exist${NC}"
fi
echo ""

# Test installation
echo "Testing installation..."
python -c "
import torch
import librosa
import gradio
print('✓ All core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
echo ""

# Summary
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo -e "${GREEN}✓ Environment configured${NC}"
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo -e "${GREEN}✓ Directory structure created${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Build Hardware:"
echo "   - See hardware/stethoscope_assembly/assembly_steps.md"
echo "   - Follow step-by-step instructions"
echo "   - Test audio recording"
echo ""
echo "2. Download & Prepare Data (if not done):"
echo "   python datasets/download_scripts/download_icbhi.py"
echo "   python datasets/preprocessing/audio_preprocessing.py --dataset icbhi"
echo "   python datasets/preprocessing/prepare_splits.py"
echo ""
echo "3. Train Model:"
echo "   python models/train.py \\"
echo "     --data_path data/splits \\"
echo "     --model_type crnn \\"
echo "     --epochs 100 \\"
echo "     --batch_size 32"
echo ""
echo "4. Run Inference:"
echo "   python models/inference.py \\"
echo "     --audio_path path/to/audio.wav \\"
echo "     --model_path models/checkpoints/best_model.pth"
echo ""
echo "5. Launch Web Interface:"
echo "   python webapp/app_gradio.py"
echo "   Open browser to http://localhost:7860"
echo ""
echo "Documentation:"
echo "   - README.md - Project overview"
echo "   - docs/dataset_description.md - Dataset details"
echo "   - hardware/stethoscope_assembly/assembly_steps.md - Hardware guide"
echo ""
echo "=================================="
echo "Happy Building! 🩺🤖"
echo "=================================="
