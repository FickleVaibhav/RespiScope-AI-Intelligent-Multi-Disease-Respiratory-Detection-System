"""
Prepare train/validation/test splits for RespiScope-AI
Implements patient-level splitting to prevent data leakage
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import get_config


def create_patient_level_splits(metadata_df, 
                                train_ratio=0.7, 
                                val_ratio=0.15, 
                                test_ratio=0.15,
                                random_state=42):
    """
    Create train/val/test splits at patient level
    
    Args:
        metadata_df: DataFrame with patient and class information
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Get unique patients
    patients = metadata_df['patient_id'].unique()
    n_patients = len(patients)
    
    print(f"Total unique patients: {n_patients}")
    
    # Get patient-level diagnoses (majority diagnosis per patient)
    patient_classes = []
    for patient_id in patients:
        patient_data = metadata_df[metadata_df['patient_id'] == patient_id]
        majority_class = patient_data['class'].mode()[0]
        patient_classes.append(majority_class)
    
    patient_classes = np.array(patient_classes)
    
    # First split: separate test set
    train_val_patients, test_patients, train_val_classes, test_classes = train_test_split(
        patients,
        patient_classes,
        test_size=test_ratio,
        stratify=patient_classes,
        random_state=random_state
    )
    
    # Second split: separate train and validation
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_patients, val_patients, train_classes, val_classes = train_test_split(
        train_val_patients,
        train_val_classes,
        test_size=relative_val_ratio,
        stratify=train_val_classes,
        random_state=random_state
    )
    
    # Create dataframes for each split
    train_df = metadata_df[metadata_df['patient_id'].isin(train_patients)].copy()
    val_df = metadata_df[metadata_df['patient_id'].isin(val_patients)].copy()
    test_df = metadata_df[metadata_df['patient_id'].isin(test_patients)].copy()
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Print statistics
    print("\n📊 Split Statistics:")
    print(f"Train: {len(train_patients)} patients, {len(train_df)} recordings")
    print(f"Val:   {len(val_patients)} patients, {len(val_df)} recordings")
    print(f"Test:  {len(test_patients)} patients, {len(test_df)} recordings")
    
    print("\n📈 Class Distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{split_name}:")
        class_counts = split_df['class'].value_counts()
        for class_name, count in class_counts.items():
            percentage = count / len(split_df) * 100
            print(f"  {class_name:15s}: {count:3d} ({percentage:5.1f}%)")
    
    return train_df, val_df, test_df


def organize_split_files(train_df, val_df, test_df, processed_dir, output_dir):
    """
    Organize files into train/val/test directories
    
    Args:
        train_df, val_df, test_df: DataFrames with split information
        processed_dir: Directory with preprocessed files
        output_dir: Output directory for organized splits
    """
    print("\n📁 Organizing files into split directories...")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split, 'audio')
        os.makedirs(split_dir, exist_ok=True)
    
    # Copy files to appropriate directories
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_dir = os.path.join(output_dir, split_name, 'audio')
        
        print(f"Processing {split_name} split...")
        for _, row in split_df.iterrows():
            src_path = os.path.join(processed_dir, row['filename'])
            dst_path = os.path.join(split_dir, row['filename'])
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, split_name, 'labels.csv')
        split_df.to_csv(metadata_path, index=False)
        print(f"✓ {split_name} split: {len(split_df)} files")
    
    print("\n✅ File organization complete!")


def verify_splits(output_dir):
    """
    Verify split integrity
    
    Args:
        output_dir: Directory containing splits
    """
    print("\n🔍 Verifying splits...")
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        audio_dir = os.path.join(split_dir, 'audio')
        labels_file = os.path.join(split_dir, 'labels.csv')
        
        # Check if directories exist
        if not os.path.exists(split_dir):
            issues.append(f"Missing {split} directory")
            continue
        
        if not os.path.exists(audio_dir):
            issues.append(f"Missing {split}/audio directory")
            continue
        
        if not os.path.exists(labels_file):
            issues.append(f"Missing {split}/labels.csv")
            continue
        
        # Load metadata
        df = pd.read_csv(labels_file)
        
        # Count files
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.npy')]
        
        if len(audio_files) != len(df):
            issues.append(f"{split}: Mismatch between audio files ({len(audio_files)}) and labels ({len(df)})")
        
        # Check for patient overlap
        if split != 'train':
            train_labels = pd.read_csv(os.path.join(output_dir, 'train', 'labels.csv'))
            train_patients = set(train_labels['patient_id'].unique())
            split_patients = set(df['patient_id'].unique())
            
            overlap = train_patients & split_patients
            if overlap:
                issues.append(f"{split}: Patient overlap with train set: {overlap}")
    
    if issues:
        print("\n❌ Verification failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All verifications passed!")
        return True


def generate_statistics(output_dir):
    """
    Generate detailed statistics for splits
    
    Args:
        output_dir: Directory containing splits
    """
    print("\n📊 Generating detailed statistics...")
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        labels_file = os.path.join(output_dir, split, 'labels.csv')
        df = pd.read_csv(labels_file)
        
        stats[split] = {
            'total_recordings': len(df),
            'total_patients': df['patient_id'].nunique(),
            'total_duration': df['duration'].sum(),
            'class_distribution': df['class'].value_counts().to_dict(),
            'avg_duration': df['duration'].mean(),
            'equipment_distribution': df['equipment'].value_counts().to_dict() if 'equipment' in df.columns else {}
        }
    
    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} SET:")
        print(f"  Recordings: {split_stats['total_recordings']}")
        print(f"  Patients: {split_stats['total_patients']}")
        print(f"  Duration: {split_stats['total_duration']:.1f} seconds ({split_stats['total_duration']/3600:.2f} hours)")
        print(f"  Avg Duration: {split_stats['avg_duration']:.1f} seconds")
        
        print(f"\n  Class Distribution:")
        for class_name, count in split_stats['class_distribution'].items():
            percentage = count / split_stats['total_recordings'] * 100
            print(f"    {class_name:15s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\n" + "="*80)
    
    # Save statistics to JSON
    import json
    stats_file = os.path.join(output_dir, 'split_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Statistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare train/val/test splits')
    parser.add_argument('--data', type=str, default='data/processed/icbhi',
                       help='Directory with preprocessed data')
    parser.add_argument('--output', type=str, default='data/splits',
                       help='Output directory for splits')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        print("❌ Error: Ratios must sum to 1.0")
        return
    
    print("="*80)
    print("RespiScope-AI Data Split Preparation")
    print("="*80)
    print(f"Input: {args.data}")
    print(f"Output: {args.output}")
    print(f"Ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    print(f"Random Seed: {args.random_seed}")
    print("="*80)
    
    # Load metadata
    metadata_file = os.path.join(args.data, 'metadata.csv')
    
    if not os.path.exists(metadata_file):
        print(f"❌ Error: Metadata file not found: {metadata_file}")
        print("Please run preprocessing first:")
        print("  python datasets/preprocessing/audio_preprocessing.py")
        return
    
    print(f"\nLoading metadata from {metadata_file}...")
    metadata_df = pd.read_csv(metadata_file)
    print(f"✓ Loaded {len(metadata_df)} recordings")
    
    # Create splits
    train_df, val_df, test_df = create_patient_level_splits(
        metadata_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )
    
    # Organize files
    organize_split_files(train_df, val_df, test_df, args.data, args.output)
    
    # Verify splits
    verify_splits(args.output)
    
    # Generate statistics
    generate_statistics(args.output)
    
    print("\n" + "="*80)
    print("✅ Data splits prepared successfully!")
    print("\nNext step:")
    print("  python models/train.py --data_path " + args.output)
    print("="*80)


if __name__ == '__main__':
    main()
