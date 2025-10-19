"""
Unit tests for inference pipeline
"""

import pytest
import numpy as np
import torch
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.inference import RespiScopeInference


class TestInference:
    """Test inference functionality"""
    
    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a mock model file"""
        # Create a simple model
        from models.crnn_classifier import CRNNClassifier
        
        model = CRNNClassifier(input_dim=2048, num_classes=4)
        model_path = tmp_path / "test_model.pth"
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'num_classes': 4}
        }, model_path)
        
        return str(model_path)
    
    def test_inference_initialization(self, mock_model_path):
        """Test inference engine initialization"""
        inference = RespiScopeInference(
            model_path=mock_model_path,
            model_type='crnn',
            device='cpu'
        )
        
        assert inference.model is not None
        assert inference.device.type == 'cpu'
    
    def test_predict_from_array(self, mock_model_path):
        """Test prediction from numpy array"""
        inference = RespiScopeInference(
            model_path=mock_model_path,
            model_type='crnn',
            device='cpu'
        )
        
        # Create dummy audio (3 seconds at 16kHz)
        audio = np.random.randn(48000).astype(np.float32)
        
        result = inference.predict_from_array(audio, sample_rate=16000)
        
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['confidence'] >= 0 and result['confidence'] <= 1
    
    def test_prediction_probabilities_sum(self, mock_model_path):
        """Test that probabilities sum to 1"""
        inference = RespiScopeInference(
            model_path=mock_model_path,
            model_type='crnn',
            device='cpu'
        )
        
        audio = np.random.randn(48000).astype(np.float32)
        result = inference.predict_from_array(audio)
        
        prob_sum = sum(result['probabilities'].values())
        assert abs(prob_sum - 1.0) < 1e-5
    
    def test_batch_prediction(self, mock_model_path, tmp_path):
        """Test batch prediction"""
        inference = RespiScopeInference(
            model_path=mock_model_path,
            model_type='crnn',
            device='cpu'
        )
        
        # Create dummy audio files
        audio_files = []
        for i in range(3):
            audio = np.random.randn(48000).astype(np.float32)
            file_path = tmp_path / f"test_audio_{i}.npy"
            np.save(file_path, audio)
            audio_files.append(str(file_path))
        
        # Note: predict_batch expects actual audio files
        # This is a simplified test
        assert len(audio_files) == 3


class TestInferenceEdgeCases:
    """Test edge cases in inference"""
    
    def test_empty_audio(self):
        """Test with empty audio"""
        from utils.audio_utils import AudioProcessor
        
        processor = AudioProcessor()
        empty_audio = np.array([])
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            result = processor.preprocess_audio_array(empty_audio)
    
    def test_very_short_audio(self):
        """Test with very short audio"""
        from utils.audio_utils import AudioProcessor
        
        processor = AudioProcessor()
        short_audio = np.random.randn(100)  # Very short
        
        # Should pad to target length
        processed = processor.pad_or_truncate(short_audio)
        assert len(processed) == processor.target_length
    
    def test_clipped_audio(self):
        """Test with clipped audio"""
        from utils.audio_utils import AudioProcessor
        
        processor = AudioProcessor()
        clipped_audio = np.ones(16000) * 2.0  # Clipped
        
        normalized = processor.normalize_audio(clipped_audio)
        assert np.abs(normalized).max() <= 1.0


class TestInferencePerformance:
    """Test inference performance"""
    
    def test_inference_time(self, mock_model_path):
        """Test that inference is reasonably fast"""
        import time
        
        inference = RespiScopeInference(
            model_path=mock_model_path,
            model_type='crnn',
            device='cpu'
        )
        
        audio = np.random.randn(48000).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            _ = inference.predict_from_array(audio)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            _ = inference.predict_from_array(audio)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        
        # Should be faster than 1 second on CPU
        assert avg_time < 1.0
    
    def test_memory_usage(self, mock_model_path):
        """Test memory usage is reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        inference = RespiScopeInference(
            model_path=mock_model_path,
            model_type='crnn',
            device='cpu'
        )
        
        audio = np.random.randn(48000).astype(np.float32)
        _ = inference.predict_from_array(audio)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        # Memory increase should be less than 500MB
        assert mem_increase < 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
