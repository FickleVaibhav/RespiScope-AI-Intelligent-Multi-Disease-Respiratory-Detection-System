"""
PyTorch Dataset classes for RespiScope-AI
Handles loading and preprocessing of respiratory audio data
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.audio_utils import AudioProcessor, AudioAugmentor


class RespiratoryDataset(Dataset):
    """Dataset class for respiratory sound data"""
    
    def __init__(self,
                 data_dir: str,
                 metadata_file: str,
                 audio_config: dict,
                 transform: Optional[Callable] = None,
                 augment: bool = False):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing preprocessed audio files
            metadata_file: Path to metadata CSV
            audio_config: Audio configuration dictionary
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.augment = augment
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=audio_config.get('sample_rate', 16000),
            duration=audio_config.get('duration', 10.0),
            n_mels=audio_config.get('n_mels', 128),
            n_fft=audio_config.get('n_fft', 2048),
            hop_length=audio_config.get('hop_length', 512),
            fmin=audio_config.get('fmin', 50),
            fmax=audio_config.get('fmax', 8000),
            top_db=audio_config.get('top_db', 30)
        )
        
        # Initialize augmentor if needed
        if self.augment:
            self.augmentor = AudioAugmentor(
                sample_rate=audio_config.get('sample_rate', 16000)
            )
        
        # Create class to index mapping
        self.classes = sorted(self.metadata['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Dataset initialized with {len(self)} samples")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (audio, label)
        """
        # Get metadata
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_file = row['filename']
        audio_path = os.path.join(self.data_dir, audio_file)
        
        try:
            audio = np.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros if file can't be loaded
            audio = np.zeros(int(self.audio_processor.target_length))
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self.augmentor.augment(audio)
        
        # Apply custom transform if provided
        if self.transform:
            audio = self.transform(audio)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get label
        label_str = row['class']
        label = self.class_to_idx[label_str]
        
        return audio_tensor, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        class_counts = self.metadata['class'].value_counts()
        total = len(self.metadata)
        
        weights = []
        for cls in self.classes:
            count = class_counts.get(cls, 1)
            weight = total / (len(self.classes) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_info(self, idx):
        """Get detailed information about a sample"""
        row = self.metadata.iloc[idx]
        return {
            'filename': row['filename'],
            'patient_id': row.get('patient_id', 'unknown'),
            'class': row['class'],
            'diagnosis': row.get('diagnosis', 'unknown'),
            'duration': row.get('duration', 0)
        }


class RespiratoryDataModule:
    """Data module for easy train/val/test loading"""
    
    def __init__(self,
                 data_root: str,
                 audio_config: dict,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 augment_train: bool = True):
        """
        Initialize data module
        
        Args:
            data_root: Root directory containing splits
            audio_config: Audio configuration
            batch_size: Batch size
            num_workers: Number of worker processes
            augment_train: Whether to augment training data
        """
        self.data_root = data_root
        self.audio_config = audio_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        # Training dataset with augmentation
        self.train_dataset = RespiratoryDataset(
            data_dir=os.path.join(self.data_root, 'train', 'audio'),
            metadata_file=os.path.join(self.data_root, 'train', 'labels.csv'),
            audio_config=self.audio_config,
            augment=self.augment_train
        )
        
        # Validation dataset without augmentation
        self.val_dataset = RespiratoryDataset(
            data_dir=os.path.join(self.data_root, 'val', 'audio'),
            metadata_file=os.path.join(self.data_root, 'val', 'labels.csv'),
            audio_config=self.audio_config,
            augment=False
        )
        
        # Test dataset without augmentation
        self.test_dataset = RespiratoryDataset(
            data_dir=os.path.join(self.data_root, 'test', 'audio'),
            metadata_file=os.path.join(self.data_root, 'test', 'labels.csv'),
            audio_config=self.audio_config,
            augment=False
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Get test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_names(self):
        """Get class names"""
        return self.train_dataset.classes if self.train_dataset else []
    
    def get_class_weights(self):
        """Get class weights for loss function"""
        return self.train_dataset.get_class_weights() if self.train_dataset else None


def test_dataset():
    """Test dataset loading"""
    print("Testing dataset...")
    
    # Mock configuration
    audio_config = {
        'sample_rate': 16000,
        'duration': 10.0,
        'n_mels': 128,
        'n_fft': 2048,
        'hop_length': 512,
        'fmin': 50,
        'fmax': 8000,
        'top_db': 30
    }
    
    # Try to load dataset
    data_dir = 'data/splits'
    
    if os.path.exists(data_dir):
        # Initialize data module
        data_module = RespiratoryDataModule(
            data_root=data_dir,
            audio_config=audio_config,
            batch_size=8,
            num_workers=2
        )
        
        # Setup datasets
        data_module.setup()
        
        # Test loading a batch
        train_loader = data_module.train_dataloader()
        
        print("\nTesting batch loading...")
        for batch_idx, (audio, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Audio shape: {audio.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels}")
            
            if batch_idx >= 2:  # Test only 3 batches
                break
        
        print("\n✅ Dataset test successful!")
    else:
        print(f"⚠️ Data directory not found: {data_dir}")
        print("Please run data preprocessing first.")


if __name__ == '__main__':
    test_dataset()
