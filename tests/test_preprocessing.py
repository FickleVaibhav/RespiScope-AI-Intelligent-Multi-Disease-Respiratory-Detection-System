"""
Unit tests for audio preprocessing
"""

import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.audio_utils import AudioProcessor
from preprocessing_v2 import RobustPreprocessor


class TestAudioProcessor:
    """Test AudioProcessor class"""
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = AudioProcessor(sample_rate=22050, duration=3.0)
        assert processor.sample_rate == 22050
        assert processor.duration == 3.0
        assert processor.target_length == 66150  # 22050 * 3
    
    def test_normalize_audio(self):
        """Test audio normalization"""
        processor = AudioProcessor()
        
        # Test with various amplitudes
        audio = np.random.randn(16000) * 2.0
        normalized = processor.normalize_audio(audio)
        
        assert np.abs(normalized).max() <= 1.0
        assert normalized.dtype == audio.dtype
    
    def test_pad_or_truncate(self):
        """Test padding and truncation"""
        processor = AudioProcessor(sample_rate=16000, duration=3.0)
        target_length = 48000
        
        # Test truncation
        long_audio = np.random.randn(60000)
        truncated = processor.pad_or_truncate(long_audio)
        assert len(truncated) == target_length
        
        # Test padding
        short_audio = np.random.randn(30000)
        padded = processor.pad_or_truncate(short_audio)
        assert len(padded) == target_length
        
        # Test exact length
        exact_audio = np.random.randn(target_length)
        result = processor.pad_or_truncate(exact_audio)
        assert len(result) == target_length
        assert np.array_equal(result, exact_audio)
    
    def test_trim_silence(self):
        """Test silence removal"""
        processor = AudioProcessor()
        
        # Create audio with silence
        silence = np.zeros(8000)
        signal = np.random.randn(8000) * 0.5
        audio_with_silence = np.concatenate([silence, signal, silence])
        
        trimmed = processor.trim_silence(audio_with_silence)
        
        # Trimmed should be shorter
        assert len(trimmed) < len(audio_with_silence)
        # Should still have signal
        assert len(trimmed) > 0
    
    def test_extract_mel_spectrogram(self):
        """Test mel-spectrogram extraction"""
        processor = AudioProcessor(sample_rate=22050, duration=3.0, n_mels=96)
        
        audio = np.random.randn(66150)
        mel_spec = processor.extract_mel_spectrogram(audio)
        
        # Check shape (time, n_mels)
        assert mel_spec.ndim == 2
        assert mel_spec.shape[1] == 96
        
        # Check normalization
        assert mel_spec.min() >= 0
        assert mel_spec.max() <= 1


class TestRobustPreprocessor:
    """Test RobustPreprocessor class"""
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        config = {'sample_rate': 22050, 'duration': 3.0}
        processor = RobustPreprocessor(config)
        
        assert processor.sr == 22050
        assert processor.duration == 3.0
        assert processor.clip_samples == 66150
    
    def test_quality_check(self):
        """Test audio quality checking"""
        processor = RobustPreprocessor()
        
        # Good quality audio
        good_audio = np.random.randn(16000) * 0.1
        quality_pass, metrics = processor.check_quality(good_audio)
        
        assert 'snr_db' in metrics
        assert 'clipping_ratio' in metrics
        assert 'silence_ratio' in metrics
        assert 'rms' in metrics
        
        # Low quality audio (clipped)
        bad_audio = np.ones(16000) * 1.5  # Heavily clipped
        quality_pass, metrics = processor.check_quality(bad_audio)
        
        assert metrics['clipping_ratio'] > 0.5
    
    def test_extract_clips(self):
        """Test clip extraction with overlap"""
        processor = RobustPreprocessor({'sample_rate': 22050, 'duration': 3.0, 'hop_duration': 1.5})
        
        # Audio longer than clip duration
        audio = np.random.randn(int(22050 * 6))  # 6 seconds
        clips = processor.extract_clips(audio)
        
        # Should have multiple overlapping clips
        assert len(clips) > 1
        
        # Each clip should have correct length
        for clip in clips:
            assert len(clip) == processor.clip_samples
    
    def test_rms_normalization(self):
        """Test RMS normalization"""
        processor = RobustPreprocessor({'target_rms_db': -20})
        
        audio = np.random.randn(16000) * 0.01
        normalized = processor.normalize_rms(audio)
        
        # Check RMS is close to target
        target_rms = 10 ** (-20 / 20)
        actual_rms = np.sqrt(np.mean(normalized ** 2))
        
        assert abs(actual_rms - target_rms) < 0.01


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_audio(self):
        """Test handling of empty audio"""
        processor = AudioProcessor()
        
        empty_audio = np.array([])
        
        # Should handle gracefully
        try:
            padded = processor.pad_or_truncate(empty_audio)
            assert len(padded) == processor.target_length
        except:
            pytest.fail("Failed to handle empty audio")
    
    def test_silent_audio(self):
        """Test handling of completely silent audio"""
        processor = AudioProcessor()
        
        silent_audio = np.zeros(16000)
        
        # Normalization should not crash
        normalized = processor.normalize_audio(silent_audio)
        assert np.all(normalized == 0)
    
    def test_nan_audio(self):
        """Test handling of NaN values"""
        processor = AudioProcessor()
        
        nan_audio = np.array([1.0, 2.0, np.nan, 3.0])
        
        # Should detect or handle NaN
        assert np.isnan(nan_audio).any()
    
    def test_very_short_audio(self):
        """Test handling of very short audio clips"""
        processor = RobustPreprocessor({'sample_rate': 22050, 'duration': 3.0})
        
        short_audio = np.random.randn(1000)  # Very short
        clips = processor.extract_clips(short_audio)
        
        # Should still produce one clip (padded)
        assert len(clips) == 1
        assert len(clips[0]) == processor.clip_samples


def test_preprocessing_pipeline():
    """Integration test for complete preprocessing"""
    processor = RobustPreprocessor({
        'sample_rate': 22050,
        'duration': 3.0,
        'n_mels': 96
    })
    
    # Simulate real audio
    audio = np.random.randn(int(22050 * 5))  # 5 seconds
    
    # Add some silence
    audio[:5000] = 0
    audio[-5000:] = 0
    
    # Extract clips
    clips = processor.extract_clips(audio)
    
    # Extract spectrograms
    spectrograms = []
    for clip in clips:
        spec = processor.compute_mel_spectrogram(clip)
        spectrograms.append(spec)
    
    # Verify
    assert len(clips) > 0
    assert len(spectrograms) == len(clips)
    
    for spec in spectrograms:
        assert spec.ndim == 2
        assert spec.shape[1] == 96


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
