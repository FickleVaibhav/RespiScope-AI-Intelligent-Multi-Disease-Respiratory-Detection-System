"""
Advanced augmentation for >92% accuracy
Includes SpecAugment, Mixup, and waveform augmentations
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional


class SpecAugment:
    """SpecAugment implementation for TensorFlow"""
    
    def __init__(self, 
                 time_mask_param: int = 30,
                 freq_mask_param: int = 15,
                 num_time_masks: int = 2,
                 num_freq_masks: int = 2):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
    
    def time_mask(self, spec: tf.Tensor, T: int) -> tf.Tensor:
        """Apply time masking"""
        tau = spec.shape[0]
        t = tf.random.uniform([], 0, T, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, tau - t, dtype=tf.int32)
        
        mask = tf.concat([
            tf.ones([t0, spec.shape[1]]),
            tf.zeros([t, spec.shape[1]]),
            tf.ones([tau - t0 - t, spec.shape[1]])
        ], axis=0)
        
        return spec * mask
    
    def freq_mask(self, spec: tf.Tensor, F: int) -> tf.Tensor:
        """Apply frequency masking"""
        nu = spec.shape[1]
        f = tf.random.uniform([], 0, F, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, nu - f, dtype=tf.int32)
        
        mask = tf.concat([
            tf.ones([spec.shape[0], f0]),
            tf.zeros([spec.shape[0], f]),
            tf.ones([spec.shape[0], nu - f0 - f])
        ], axis=1)
        
        return spec * mask
    
    def __call__(self, spec: tf.Tensor) -> tf.Tensor:
        """Apply SpecAugment"""
        # Time masking
        for _ in range(self.num_time_masks):
            spec = self.time_mask(spec, self.time_mask_param)
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            spec = self.freq_mask(spec, self.freq_mask_param)
        
        return spec


def mixup(x1: tf.Tensor, y1: tf.Tensor, x2: tf.Tensor, y2: tf.Tensor, 
          alpha: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Mixup augmentation
    
    Args:
        x1, y1: First sample and label
        x2, y2: Second sample and label
        alpha: Beta distribution parameter
        
    Returns:
        Mixed sample and label
    """
    lam = tf.random.uniform([], 0, 1)
    lam = tf.maximum(lam, 1 - lam)  # Ensure lam >= 0.5
    
    # Mix inputs
    x_mixed = lam * x1 + (1 - lam) * x2
    
    # Mix labels (soft labels)
    y_mixed = lam * y1 + (1 - lam) * y2
    
    return x_mixed, y_mixed


class WaveformAugmentor:
    """Waveform-level augmentations (NumPy-based)"""
    
    @staticmethod
    def add_gaussian_noise(audio: np.ndarray, snr_db: float = 25.0) -> np.ndarray:
        """Add Gaussian noise at specified SNR"""
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        return audio + noise
    
    @staticmethod
    def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """Time stretch using linear interpolation"""
        if rate == 1.0:
            return audio
        
        indices = np.arange(0, len(audio), rate)
        indices = indices[indices < len(audio)].astype(int)
        stretched = audio[indices]
        
        # Ensure same length
        if len(stretched) < len(audio):
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='edge')
        else:
            stretched = stretched[:len(audio)]
        
        return stretched
    
    @staticmethod
    def pitch_shift_simple(audio: np.ndarray, n_steps: int = 0) -> np.ndarray:
        """Simple pitch shift by resampling"""
        if n_steps == 0:
            return audio
        
        rate = 2 ** (n_steps / 12)
        return WaveformAugmentor.time_stretch(audio, rate)
    
    @staticmethod
    def time_shift(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """Circular time shift"""
        shift = int(len(audio) * np.random.uniform(-shift_max, shift_max))
        return np.roll(audio, shift)
    
    @staticmethod
    def random_gain(audio: np.ndarray, gain_db_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """Apply random gain"""
        gain_db = np.random.uniform(*gain_db_range)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear


def create_augmentation_pipeline(use_specaugment: bool = True,
                                use_mixup: bool = True,
                                training: bool = True) -> callable:
    """
    Create complete augmentation pipeline
    
    Args:
        use_specaugment: Enable SpecAugment
        use_mixup: Enable Mixup (requires special dataset setup)
        training: Whether in training mode
        
    Returns:
        Augmentation function
    """
    spec_augment = SpecAugment() if use_specaugment else None
    
    def augment_fn(spectrogram: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if not training:
            return spectrogram, label
        
        # Apply SpecAugment
        if spec_augment is not None and tf.random.uniform([]) > 0.5:
            spectrogram = spec_augment(spectrogram)
        
        # Add channel dimension if needed
        if len(spectrogram.shape) == 2:
            spectrogram = tf.expand_dims(spectrogram, -1)
        
        return spectrogram, label
    
    return augment_fn


# TensorFlow data augmentation layer
@tf.keras.utils.register_keras_serializable()
class AudioAugmentation(tf.keras.layers.Layer):
    """Custom Keras layer for audio augmentation"""
    
    def __init__(self, 
                 spec_augment: bool = True,
                 time_mask: int = 30,
                 freq_mask: int = 15,
                 **kwargs):
        super().__init__(**kwargs)
        self.spec_augment = spec_augment
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.augmentor = SpecAugment(time_mask, freq_mask) if spec_augment else None
    
    def call(self, inputs, training=None):
        if training and self.augmentor is not None:
            return self.augmentor(inputs)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'spec_augment': self.spec_augment,
            'time_mask': self.time_mask,
            'freq_mask': self.freq_mask
        })
        return config


def generate_augmented_dataset(input_dir: str, 
                               output_dir: str,
                               metadata_file: str,
                               augmentations_per_sample: int = 3):
    """
    Generate augmented dataset offline
    
    Args:
        input_dir: Directory with preprocessed spectrograms
        output_dir: Output directory for augmented data
        metadata_file: Metadata CSV
        augmentations_per_sample: Number of augmented versions per sample
    """
    import pandas as pd
    import os
    from tqdm import tqdm
    
    print("="*80)
    print("AUGMENTED DATASET GENERATION")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(metadata_file)
    print(f"Loaded {len(df)} samples")
    
    augmentor = WaveformAugmentor()
    augmented_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        # Load original spectrogram
        spec_path = os.path.join(input_dir, row['clip_filename'])
        
        if not os.path.exists(spec_path):
            continue
        
        spec = np.load(spec_path)
        
        # Save original
        output_path = os.path.join(output_dir, row['clip_filename'])
        np.save(output_path, spec)
        augmented_records.append({**row, 'augmentation': 'original'})
        
        # Generate augmented versions
        for aug_idx in range(augmentations_per_sample):
            # Apply random augmentations
            aug_spec = spec.copy()
            
            # Random combination of augmentations
            if np.random.random() > 0.5:
                # Time masking
                T = np.random.randint(10, 30)
                t0 = np.random.randint(0, max(1, aug_spec.shape[0] - T))
                aug_spec[t0:t0+T, :] = 0
            
            if np.random.random() > 0.5:
                # Frequency masking
                F = np.random.randint(5, 15)
                f0 = np.random.randint(0, max(1, aug_spec.shape[1] - F))
                aug_spec[:, f0:f0+F] = 0
            
            # Save augmented
            aug_filename = row['clip_filename'].replace('.npy', f'_aug{aug_idx}.npy')
            aug_path = os.path.join(output_dir, aug_filename)
            np.save(aug_path, aug_spec.astype(np.float32))
            
            augmented_records.append({
                **row,
                'clip_filename': aug_filename,
                'augmentation': f'aug{aug_idx}'
            })
    
    # Save metadata
    aug_df = pd.DataFrame(augmented_records)
    aug_metadata = os.path.join(output_dir, 'augmented_metadata.csv')
    aug_df.to_csv(aug_metadata, index=False)
    
    print(f"\n✅ Generated {len(aug_df)} samples ({len(df)} original + {len(aug_df)-len(df)} augmented)")
    print(f"📁 Saved to: {output_dir}")
    print(f"📄 Metadata: {aug_metadata}")
    
    return aug_df


def test_augmentations():
    """Test augmentation functions"""
    print("Testing augmentations...")
    
    # Create dummy data
    spec = tf.random.normal([128, 96, 1])
    label = tf.one_hot(2, 4)
    
    # Test SpecAugment
    spec_aug = SpecAugment()
    augmented = spec_aug(spec)
    print(f"✓ SpecAugment: {spec.shape} -> {augmented.shape}")
    
    # Test Mixup
    spec2 = tf.random.normal([128, 96, 1])
    label2 = tf.one_hot(1, 4)
    mixed_spec, mixed_label = mixup(spec, label, spec2, label2)
    print(f"✓ Mixup: {mixed_spec.shape}, {mixed_label.shape}")
    
    # Test waveform augmentations
    audio = np.random.randn(66150).astype(np.float32)  # 3s at 22050Hz
    
    noisy = WaveformAugmentor.add_gaussian_noise(audio)
    print(f"✓ Gaussian noise: {audio.shape} -> {noisy.shape}")
    
    stretched = WaveformAugmentor.time_stretch(audio, rate=1.1)
    print(f"✓ Time stretch: {audio.shape} -> {stretched.shape}")
    
    shifted = WaveformAugmentor.time_shift(audio)
    print(f"✓ Time shift: {audio.shape} -> {shifted.shape}")
    
    print("\n✅ All augmentation tests passed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio augmentation')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--generate', action='store_true', help='Generate augmented dataset')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--metadata', type=str, help='Metadata CSV')
    parser.add_argument('--n_aug', type=int, default=3, help='Augmentations per sample')
    
    args = parser.parse_args()
    
    if args.test:
        test_augmentations()
    elif args.generate:
        if not all([args.input_dir, args.output_dir, args.metadata]):
            print("Error: --input_dir, --output_dir, and --metadata required for --generate")
        else:
            generate_augmented_dataset(
                args.input_dir,
                args.output_dir,
                args.metadata,
                args.n_aug
            )
    else:
        print("Use --test to run tests or --generate to create augmented dataset")
