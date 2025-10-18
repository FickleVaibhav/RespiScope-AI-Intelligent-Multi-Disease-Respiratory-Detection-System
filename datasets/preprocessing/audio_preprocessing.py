"""
Audio preprocessing for RespiScope-AI
Handles resampling, normalization, and feature extraction
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.audio_utils import AudioProcessor
from utils.config import get_config


# ICBHI diagnosis mapping to our classes
DIAGNOSIS_MAPPING = {
    'COPD': 'COPD',
    'Bronchiectasis': 'Bronchitis',
    'Pneumonia': 'Pneumonia',
    'URTI': 'Healthy',  # Upper respiratory tract infection (mild)
    'Bronchiolitis': 'Bronchitis',
    'Asthma': 'Asthma',
    'LRTI': 'Pneumonia',  # Lower respiratory tract infection
    'Healthy': 'Healthy'
}


def parse_icbhi_annotations(txt_file: str):
    """Parse ICBHI annotation file"""
    annotations = []
    
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                start, end, crackles, wheezes = parts[:4]
                annotations.append({
                    'start': float(start),
                    'end': float(end),
                    'crackles': int(crackles),
                    'wheezes': int(wheezes)
                })
    
    return annotations


def parse_icbhi_filename(filename: str):
    """Extract patient info from ICBHI filename"""
    # Format: PatientID_RecordingIndex_ChestLocation_AcquisitionMode_RecordingEquipment
    # Example: 101_1b1_Al_sc_Meditron.wav
    
    parts = filename.replace('.wav', '').split('_')
    
    return {
        'patient_id': parts[0],
        'recording_index': parts[1] if len(parts) > 1 else '',
        'chest_location': parts[2] if len(parts) > 2 else '',
        'acquisition_mode': parts[3] if len(parts) > 3 else '',
        'equipment': parts[4] if len(parts) > 4 else ''
    }


def load_icbhi_metadata(data_dir: str):
    """Load ICBHI patient metadata"""
    metadata_file = os.path.join(data_dir, 'patient_diagnosis.csv')
    
    if not os.path.exists(metadata_file):
        # Try alternative name
        metadata_file = os.path.join(data_dir, 'demographic_info.txt')
    
    if not os.path.exists(metadata_file):
        print("⚠️ Warning: Metadata file not found. Creating from audio files...")
        return None
    
    # Read metadata
    try:
        if metadata_file.endswith('.csv'):
            df = pd.read_csv(metadata_file)
        else:
            df = pd.read_csv(metadata_file, delimiter='\t')
        
        return df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def preprocess_icbhi_dataset(input_dir: str, output_dir: str, config):
    """Preprocess ICBHI dataset"""
    print("\n🔄 Preprocessing ICBHI dataset...")
    
    audio_dir = os.path.join(input_dir, 'audio_and_txt_files')
    if not os.path.exists(audio_dir):
        audio_dir = input_dir
    
    # Get all audio files
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load metadata
    metadata = load_icbhi_metadata(input_dir)
    
    # Initialize audio processor
    processor = AudioProcessor(
        sample_rate=config.audio.sample_rate,
        duration=config.audio.duration,
        n_mels=config.audio.n_mels,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        fmin=config.audio.fmin,
        fmax=config.audio.fmax,
        top_db=config.audio.top_db
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each audio file
    processed_data = []
    
    for audio_file in tqdm(audio_files, desc="Processing"):
        try:
            audio_path = os.path.join(audio_dir, audio_file)
            
            # Parse filename
            file_info = parse_icbhi_filename(audio_file)
            patient_id = file_info['patient_id']
            
            # Get diagnosis from metadata
            diagnosis = 'Healthy'  # Default
            if metadata is not None:
                patient_row = metadata[metadata['Patient number'] == int(patient_id)]
                if not patient_row.empty:
                    diagnosis = patient_row['Diagnosis'].values[0]
            
            # Map to our classes
            mapped_class = DIAGNOSIS_MAPPING.get(diagnosis, 'Healthy')
            
            # Preprocess audio
            audio = processor.preprocess(audio_path, trim=True, normalize=True)
            
            # Save preprocessed audio
            output_filename = audio_file.replace('.wav', '_processed.npy')
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, audio)
            
            # Store metadata
            processed_data.append({
                'filename': output_filename,
                'original_file': audio_file,
                'patient_id': patient_id,
                'diagnosis': diagnosis,
                'class': mapped_class,
                'chest_location': file_info['chest_location'],
                'equipment': file_info['equipment'],
                'duration': len(audio) / config.audio.sample_rate
            })
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Save metadata
    metadata_df = pd.DataFrame(processed_data)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    # Print statistics
    print(f"\n✅ Preprocessing complete!")
    print(f"Processed: {len(processed_data)} files")
    print(f"Output directory: {output_dir}")
    
    print("\n📊 Class distribution:")
    class_counts = metadata_df['class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(metadata_df)*100:.1f}%)")
    
    return metadata_df


def preprocess_coswara_dataset(input_dir: str, output_dir: str, config):
    """Preprocess Coswara dataset"""
    print("\n🔄 Preprocessing Coswara dataset...")
    print("⚠️ Coswara preprocessing not yet implemented.")
    print("Please implement based on Coswara dataset structure.")
    
    # TODO: Implement Coswara preprocessing
    # The dataset has different structure with cough, breathing, and voice samples
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Preprocess respiratory audio datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['icbhi', 'coswara', 'all'],
                       help='Dataset to preprocess')
    parser.add_argument('--input_dir', type=str, default='data/raw',
                       help='Input directory containing raw data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        from utils.config import load_config
        config = load_config(args.config)
    else:
        config = get_config()
    
    print("=" * 80)
    print("RespiScope-AI Audio Preprocessing")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Sample rate: {config.audio.sample_rate} Hz")
    print(f"Duration: {config.audio.duration} seconds")
    print("=" * 80)
    
    # Process datasets
    if args.dataset == 'icbhi' or args.dataset == 'all':
        icbhi_input = os.path.join(args.input_dir, 'icbhi')
        icbhi_output = os.path.join(args.output_dir, 'icbhi')
        
        if os.path.exists(icbhi_input):
            preprocess_icbhi_dataset(icbhi_input, icbhi_output, config)
        else:
            print(f"⚠️ ICBHI dataset not found at {icbhi_input}")
            print("Please run: python datasets/download_scripts/download_icbhi.py")
    
    if args.dataset == 'coswara' or args.dataset == 'all':
        coswara_input = os.path.join(args.input_dir, 'coswara')
        coswara_output = os.path.join(args.output_dir, 'coswara')
        
        if os.path.exists(coswara_input):
            preprocess_coswara_dataset(coswara_input, coswara_output, config)
        else:
            print(f"⚠️ Coswara dataset not found at {coswara_input}")
    
    print("\n" + "=" * 80)
    print("✅ Preprocessing complete!")
    print("\nNext step:")
    print("Run: python datasets/preprocessing/prepare_splits.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
