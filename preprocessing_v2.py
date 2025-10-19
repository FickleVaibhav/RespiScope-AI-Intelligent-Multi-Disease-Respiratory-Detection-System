"""
Robust preprocessing pipeline for >92% accuracy
Target: 22.05kHz, 3-second clips, log-mel spectrograms
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION
TARGET_SR = 22050  # Hz
CLIP_DURATION = 3.0  # seconds
HOP_DURATION = 1.5  # seconds (50% overlap)
N_MELS = 96
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 50
FMAX = 8000
TOP_DB = 20  # For silence removal
TARGET_RMS_DB = -20  # Target loudness


class RobustPreprocessor:
    """Production-grade audio preprocessing"""
    
    def __init__(self, config: dict = None):
        self.sr = config.get('sample_rate', TARGET_SR) if config else TARGET_SR
        self.duration = config.get('duration', CLIP_DURATION) if config else CLIP_DURATION
        self.hop_duration = config.get('hop_duration', HOP_DURATION) if config else HOP_DURATION
        self.n_mels = config.get('n_mels', N_MELS) if config else N_MELS
        self.n_fft = config.get('n_fft', N_FFT) if config else N_FFT
        self.hop_length = config.get('hop_length', HOP_LENGTH) if config else HOP_LENGTH
        self.fmin = config.get('fmin', FMIN) if config else FMIN
        self.fmax = config.get('fmax', FMAX) if config else FMAX
        self.top_db = config.get('top_db', TOP_DB) if config else TOP_DB
        self.target_rms_db = config.get('target_rms_db', TARGET_RMS_DB) if config else TARGET_RMS_DB
        
        self.clip_samples = int(self.sr * self.duration)
        self.hop_samples = int(self.sr * self.hop_duration)
        
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio with robust error handling"""
        try:
            # Use librosa with kaiser_best resampling (highest quality)
            audio, sr = librosa.load(filepath, sr=self.sr, res_type='kaiser_best', mono=True)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load {filepath}: {e}")
    
    def check_quality(self, audio: np.ndarray) -> Tuple[bool, dict]:
        """Check audio quality and compute metrics"""
        metrics = {}
        
        # Compute SNR estimate
        signal_power = np.mean(audio ** 2)
        noise_floor = np.percentile(np.abs(audio), 10) ** 2
        snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))
        metrics['snr_db'] = float(snr_db)
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        metrics['clipping_ratio'] = float(clipping_ratio)
        
        # Check for silence
        silence_ratio = np.sum(np.abs(audio) < 0.01) / len(audio)
        metrics['silence_ratio'] = float(silence_ratio)
        
        # Compute RMS
        rms = np.sqrt(np.mean(audio ** 2))
        metrics['rms'] = float(rms)
        
        # Quality checks
        quality_pass = (
            snr_db > 15 and  # Minimum SNR
            clipping_ratio < 0.01 and  # Less than 1% clipping
            silence_ratio < 0.9 and  # Less than 90% silence
            rms > 0.001  # Not completely silent
        )
        
        return quality_pass, metrics
    
    def remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove silence from beginning and end"""
        try:
            audio_trimmed, _ = librosa.effects.trim(
                audio,
                top_db=self.top_db,
                frame_length=2048,
                hop_length=512
            )
            # Ensure we still have audio left
            if len(audio_trimmed) < self.sr * 0.5:  # Less than 0.5s
                return audio  # Return original if trimming was too aggressive
            return audio_trimmed
        except:
            return audio
    
    def normalize_rms(self, audio: np.ndarray) -> np.ndarray:
        """Normalize to target RMS level"""
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms < 1e-6:  # Avoid division by zero
            return audio
        
        target_rms = 10 ** (self.target_rms_db / 20)
        scaling_factor = target_rms / current_rms
        audio_normalized = audio * scaling_factor
        
        # Prevent clipping
        max_val = np.abs(audio_normalized).max()
        if max_val > 0.99:
            audio_normalized = audio_normalized * (0.99 / max_val)
        
        return audio_normalized
    
    def extract_clips(self, audio: np.ndarray) -> List[np.ndarray]:
        """Extract overlapping 3-second clips"""
        clips = []
        
        if len(audio) < self.clip_samples:
            # Pad if audio is shorter than clip duration
            audio_padded = np.pad(audio, (0, self.clip_samples - len(audio)), mode='constant')
            clips.append(audio_padded)
        else:
            # Extract overlapping clips
            start = 0
            while start + self.clip_samples <= len(audio):
                clip = audio[start:start + self.clip_samples]
                clips.append(clip)
                start += self.hop_samples
            
            # Add final clip if there's remaining audio
            if start < len(audio):
                final_clip = audio[-self.clip_samples:]
                clips.append(final_clip)
        
        return clips
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram"""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db + 80) / 80
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
        
        return mel_spec_norm.T  # Shape: (time, n_mels)
    
    def compute_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """Compute MFCC features (optional)"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Add delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return mfcc_features.T  # Shape: (time, n_mfcc * 3)
    
    def process_file(self, filepath: str, extract_features: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], dict]:
        """
        Complete preprocessing pipeline for one file
        
        Returns:
            clips: List of preprocessed audio clips
            spectrograms: List of mel spectrograms (if extract_features=True)
            metadata: Quality metrics and processing info
        """
        # Load audio
        audio, sr = self.load_audio(filepath)
        
        # Check quality
        quality_pass, metrics = self.check_quality(audio)
        
        # Remove silence
        audio_trimmed = self.remove_silence(audio)
        
        # Normalize
        audio_normalized = self.normalize_rms(audio_trimmed)
        
        # Extract clips
        clips = self.extract_clips(audio_normalized)
        
        # Compute spectrograms
        spectrograms = []
        if extract_features:
            for clip in clips:
                mel_spec = self.compute_mel_spectrogram(clip)
                spectrograms.append(mel_spec)
        
        # Metadata
        metadata = {
            'filename': os.path.basename(filepath),
            'quality_pass': quality_pass,
            'num_clips': len(clips),
            'original_duration': len(audio) / sr,
            'trimmed_duration': len(audio_trimmed) / sr,
            **metrics
        }
        
        return clips, spectrograms, metadata


def process_dataset(input_dir: str, output_dir: str, metadata_file: str, config: dict = None):
    """Process entire dataset"""
    
    print("="*80)
    print("ROBUST PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target SR: {TARGET_SR} Hz")
    print(f"Clip Duration: {CLIP_DURATION}s")
    print(f"Overlap: {HOP_DURATION}s")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = RobustPreprocessor(config)
    
    # Load metadata
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    df = pd.read_csv(metadata_file)
    print(f"\nLoaded metadata: {len(df)} recordings")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'waveforms'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'spectrograms'), exist_ok=True)
    
    # Process each file
    processed_records = []
    failed_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            filepath = os.path.join(input_dir, row['filename'])
            
            if not os.path.exists(filepath):
                print(f"⚠️ File not found: {filepath}")
                continue
            
            # Process file
            clips, spectrograms, metadata = preprocessor.process_file(filepath, extract_features=True)
            
            # Save clips and spectrograms
            for clip_idx, (clip, spec) in enumerate(zip(clips, spectrograms)):
                base_name = Path(row['filename']).stem
                clip_name = f"{base_name}_clip{clip_idx}"
                
                # Save waveform
                waveform_path = os.path.join(output_dir, 'waveforms', f"{clip_name}.npy")
                np.save(waveform_path, clip.astype(np.float32))
                
                # Save spectrogram
                spec_path = os.path.join(output_dir, 'spectrograms', f"{clip_name}.npy")
                np.save(spec_path, spec.astype(np.float32))
                
                # Record metadata
                processed_records.append({
                    'clip_filename': f"{clip_name}.npy",
                    'original_file': row['filename'],
                    'clip_index': clip_idx,
                    'patient_id': row.get('patient_id', 'unknown'),
                    'class': row['class'],
                    'diagnosis': row.get('diagnosis', row['class']),
                    'split': row.get('split', 'unknown'),
                    **metadata
                })
        
        except Exception as e:
            print(f"❌ Error processing {row['filename']}: {e}")
            failed_files.append(row['filename'])
            continue
    
    # Save processed metadata
    processed_df = pd.DataFrame(processed_records)
    output_metadata = os.path.join(output_dir, 'processed_metadata.csv')
    processed_df.to_csv(output_metadata, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"✅ Processed: {len(processed_records)} clips from {len(df) - len(failed_files)} recordings")
    print(f"❌ Failed: {len(failed_files)} recordings")
    print(f"\n📊 Clips per class:")
    print(processed_df.groupby('class').size())
    print(f"\n📁 Output saved to: {output_dir}")
    print(f"📄 Metadata: {output_metadata}")
    
    # Quality report
    quality_passed = processed_df['quality_pass'].sum()
    print(f"\n🔍 Quality Check:")
    print(f"  Passed: {quality_passed}/{len(processed_df)} ({quality_passed/len(processed_df)*100:.1f}%)")
    print(f"  Avg SNR: {processed_df['snr_db'].mean():.1f} dB")
    print(f"  Avg Clipping: {processed_df['clipping_ratio'].mean()*100:.2f}%")
    
    return processed_df


def main():
    parser = argparse.ArgumentParser(description='Robust preprocessing for RespiScope-AI')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with raw audio')
    parser.add_argument('--metadata', type=str, required=True, help='Metadata CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--sr', type=int, default=TARGET_SR, help='Target sample rate')
    parser.add_argument('--duration', type=float, default=CLIP_DURATION, help='Clip duration in seconds')
    
    args = parser.parse_args()
    
    config = {
        'sample_rate': args.sr,
        'duration': args.duration,
        'hop_duration': args.duration / 2,
        'n_mels': N_MELS,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'fmin': FMIN,
        'fmax': FMAX,
        'top_db': TOP_DB,
        'target_rms_db': TARGET_RMS_DB
    }
    
    process_dataset(args.input_dir, args.output_dir, args.metadata, config)


if __name__ == '__main__':
    main()
