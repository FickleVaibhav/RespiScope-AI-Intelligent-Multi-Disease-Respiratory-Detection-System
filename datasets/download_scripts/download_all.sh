#!/bin/bash

# Download all respiratory datasets for RespiScope-AI

set -e

echo "=========================================="
echo "RespiScope-AI Dataset Download Script"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create data directories
mkdir -p data/raw/icbhi
mkdir -p data/raw/coswara
mkdir -p data/processed
mkdir -p data/splits

echo "✓ Data directories created"
echo ""

# Download ICBHI dataset
echo "=========================================="
echo "Downloading ICBHI 2017 Dataset"
echo "=========================================="
python3 datasets/download_scripts/download_icbhi.py --output_dir data/raw/icbhi

if [ $? -eq 0 ]; then
    echo "✓ ICBHI dataset downloaded successfully"
else
    echo "✗ ICBHI download failed"
fi
echo ""

# Download Coswara dataset (optional)
read -p "Download Coswara dataset? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "=========================================="
    echo "Downloading Coswara Dataset"
    echo "=========================================="
    python3 datasets/download_scripts/download_coswara.py --output_dir data/raw/coswara
    
    if [ $? -eq 0 ]; then
        echo "✓ Coswara dataset downloaded successfully"
    else
        echo "✗ Coswara download failed"
    fi
fi
echo ""

echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run preprocessing:"
echo "   python3 datasets/preprocessing/audio_preprocessing.py"
echo ""
echo "2. Prepare train/val/test splits:"
echo "   python3 datasets/preprocessing/prepare_splits.py"
echo ""
echo "3. Start training:"
echo "   python3 models/train.py --data_path data/splits"
echo ""
