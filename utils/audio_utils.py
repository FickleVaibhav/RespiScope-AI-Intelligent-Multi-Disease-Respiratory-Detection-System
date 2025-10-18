"""
Audio utility functions for RespiScope-AI
Handles audio loading, preprocessing, and feature extraction
"""

import librosa
import numpy as np
import soundfile as sf
import torch
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class AudioProcessor:
    """Audio preprocessing and feature extraction"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 duration: float = 10.0,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 fmin: int = 50,
                 fmax: int = 8000,
                 top_db: int = 30):
        """
        Initialize AudioProcessor
        
        Args:
            sample_rate: Target sampling rate
            duration: Maximum audio duration in seconds
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            fmin: Minimum frequency
            fmax: Maximum frequency
            top_db: Threshold for silence removal
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.top_db = top_db
        self.target_length = int(sample_rate * duration)
    
    def load_audio(self, 
                   audio_path: str,
                   offset: float = 0.0,
                   duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            offset: Start reading after this time (in seconds)
            duration: Only load up to this much audio (in seconds)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=offset,
                duration=duration,
                mono=True
            )
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {str(e)}")
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio
        """
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        return audio
    
    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove silence from beginning and end of audio
        
        Args:
            audio: Input audio array
            
        Returns:
            Trimmed audio
        """
        audio_trimmed, _ = librosa.effects.trim(
            audio,
            top_db=self.top_db
        )
        return audio_trimmed
    
    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad or truncate audio to target length
        
        Args:
            audio: Input audio array
            
        Returns:
            Audio with target length
        """
        if len(audio) > self.target_length:
            # Truncate
            audio = audio[:self.target_length]
        elif len(audio) < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio
        
        Args:
            audio: Input audio array
            
        Returns:
            Mel-spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """
        Extract MFCC features from audio
        
        Args:
            audio: Input audio array
            n_mfcc: Number of MFCCs to extract
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Add delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return mfcc_features
    
    def preprocess(self, 
                   audio_path: str,
                   trim: bool = True,
                   normalize: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            audio_path: Path to audio file
            trim: Whether to trim silence
            normalize: Whether to normalize audio
            
        Returns:
            Preprocessed audio
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Trim silence
        if trim:
            audio = self.trim_silence(audio)
        
        # Normalize
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Pad or truncate
        audio = self.pad_or_truncate(audio)
        
        return audio
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio to file
        
        Args:
            audio: Audio array to save
            output_path: Output file path
        """
        sf.write(output_path, audio, self.sample_rate)


def apply_pitch_shift(audio: np.ndarray, 
                      sr: int, 
                      n_steps: float) -> np.ndarray:
    """
    Apply pitch shifting to audio
    
    Args:
        audio: Input audio array
        sr: Sample rate
        n_steps: Number of semitones to shift (can be fractional)
        
    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def apply_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Apply time stretching to audio
    
    Args:
        audio: Input audio array
        rate: Stretch factor (> 1.0 speeds up, < 1.0 slows down)
        
    Returns:
        Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """
    Add random noise to audio
    
    Args:
        audio: Input audio array
        noise_level: Standard deviation of noise
        
    Returns:
        Audio with added noise
    """
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def add_noise_snr(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add noise with specified SNR
    
    Args:
        audio: Input audio array
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Audio with added noise
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power based on SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    
    return audio + noise


def bandpass_filter(audio: np.ndarray, 
                    sr: int,
                    lowcut: float = 100,
                    highcut: float = 4000,
                    order: int = 5) -> np.ndarray:
    """
    Apply bandpass filter to audio
    
    Args:
        audio: Input audio array
        sr: Sample rate
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        order: Filter order
        
    Returns:
        Filtered audio
    """
    from scipy.signal import butter, filtfilt
    
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_audio = filtfilt(b, a, audio)
    
    return filtered_audio


def compute_audio_stats(audio: np.ndarray) -> dict:
    """
    Compute statistics of audio signal
    
    Args:
        audio: Input audio array
        
    Returns:
        Dictionary of audio statistics
    """
    stats = {
        'mean': np.mean(audio),
        'std': np.std(audio),
        'min': np.min(audio),
        'max': np.max(audio),
        'rms': np.sqrt(np.mean(audio ** 2)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
        'duration': len(audio)
    }
    
    return stats


def split_audio_into_segments(audio: np.ndarray,
                              sr: int,
                              segment_duration: float = 5.0,
                              overlap: float = 0.5) -> list:
    """
    Split audio into overlapping segments
    
    Args:
        audio: Input audio array
        sr: Sample rate
        segment_duration: Duration of each segment in seconds
        overlap: Overlap ratio (0.0 to 1.0)
        
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    # Add last segment if there's remaining audio
    if start < len(audio):
        last_segment = audio[start:]
        # Pad if necessary
        if len(last_segment) < segment_samples:
            last_segment = np.pad(
                last_segment,
                (0, segment_samples - len(last_segment)),
                mode='constant'
            )
        segments.append(last_segment)
    
    return segments


def convert_to_tensor(audio: np.ndarray,
                     normalize: bool = True) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor
    
    Args:
        audio: Input audio array
        normalize: Whether to normalize to [-1, 1]
        
    Returns:
        PyTorch tensor
    """
    if normalize and np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max()
    
    tensor = torch.from_numpy(audio).float()
    
    return tensor


class AudioAugmentor:
    """Audio augmentation pipeline"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 pitch_shift_prob: float = 0.3,
                 pitch_shift_range: tuple = (-2, 2),
                 time_stretch_prob: float = 0.3,
                 time_stretch_range: tuple = (0.9, 1.1),
                 noise_prob: float = 0.3,
                 noise_snr_range: tuple = (10, 30)):
        """
        Initialize AudioAugmentor
        
        Args:
            sample_rate: Audio sample rate
            pitch_shift_prob: Probability of applying pitch shift
            pitch_shift_range: Range of pitch shift in semitones
            time_stretch_prob: Probability of applying time stretch
            time_stretch_range: Range of time stretch factors
            noise_prob: Probability of adding noise
            noise_snr_range: Range of SNR in dB for noise
        """
        self.sample_rate = sample_rate
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
    
    def augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to audio
        
        Args:
            audio: Input audio array
            
        Returns:
            Augmented audio
        """
        # Pitch shift
        if np.random.random() < self.pitch_shift_prob:
            n_steps = np.random.uniform(*self.pitch_shift_range)
            audio = apply_pitch_shift(audio, self.sample_rate, n_steps)
        
        # Time stretch
        if np.random.random() < self.time_stretch_prob:
            rate = np.random.uniform(*self.time_stretch_range)
            audio = apply_time_stretch(audio, rate)
        
        # Add noise
        if np.random.random() < self.noise_prob:
            snr_db = np.random.uniform(*self.noise_snr_range)
            audio = add_noise_snr(audio, snr_db)
        
        return audio


if __name__ == "__main__":
    # Example usage
    processor = AudioProcessor(sample_rate=16000, duration=10.0)
    
    # Test with dummy audio
    dummy_audio = np.random.randn(16000 * 5)  # 5 seconds of audio
    
    print("Original audio shape:", dummy_audio.shape)
    
    # Preprocess
    processed = processor.pad_or_truncate(dummy_audio)
    print("Processed audio shape:", processed.shape)
    
    # Extract features
    mel_spec = processor.extract_mel_spectrogram(processed)
    print("Mel-spectrogram shape:", mel_spec.shape)
    
    mfcc = processor.extract_mfcc(processed)
    print("MFCC shape:", mfcc.shape)
    
    # Test augmentation
    augmentor = AudioAugmentor()
    augmented = augmentor.augment(dummy_audio)
    print("Augmented audio shape:", augmented.shape)
    
    print("\nAudio utilities loaded successfully!")
