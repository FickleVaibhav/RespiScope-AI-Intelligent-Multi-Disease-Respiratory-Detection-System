"""
Unit tests for model architectures
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.crnn_classifier import CRNNClassifier, RespiScopeModel
from models.transformer_classifier import TransformerClassifier


class TestCRNNClassifier:
    """Test CRNN model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = CRNNClassifier(
            input_dim=2048,
            num_classes=5,
            rnn_hidden=256,
            rnn_layers=2
        )
        
        assert model.num_classes == 5
        assert model.rnn_hidden == 256
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        
        # Test with 2D input
        x = torch.randn(4, 2048)
        output = model(x)
        
        assert output.shape == (4, 5)
        assert not torch.isnan(output).any()
    
    def test_output_range(self):
        """Test output is valid logits"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        model.eval()
        
        x = torch.randn(8, 2048)
        output = model(x)
        
        # Logits can be any value
        assert output.dtype == torch.float32
        
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 2048)
            output = model(x)
            assert output.shape == (batch_size, 5)
    
    def test_return_features(self):
        """Test feature extraction"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        
        x = torch.randn(4, 2048)
        logits, features = model(x, return_features=True)
        
        assert logits.shape == (4, 5)
        assert features.shape[0] == 4
        assert features.ndim == 2


class TestTransformerClassifier:
    """Test Transformer model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = TransformerClassifier(
            input_dim=2048,
            d_model=512,
            num_classes=5
        )
        
        assert model.num_classes == 5
        assert model.d_model == 512
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = TransformerClassifier(input_dim=2048, num_classes=5)
        
        x = torch.randn(4, 2048)
        output = model(x)
        
        assert output.shape == (4, 5)
        assert not torch.isnan(output).any()
    
    def test_positional_encoding(self):
        """Test positional encoding"""
        model = TransformerClassifier(input_dim=2048, num_classes=5)
        
        # Same input should give same output
        x = torch.ones(4, 2048)
        output1 = model(x)
        output2 = model(x)
        
        assert torch.allclose(output1, output2, atol=1e-5)


class TestModelIntegration:
    """Integration tests"""
    
    def test_model_training_mode(self):
        """Test model in training vs eval mode"""
        model = CRNNClassifier(input_dim=2048, num_classes=5, dropout=0.5)
        
        x = torch.randn(4, 2048)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Eval mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval)
    
    def test_gradient_flow(self):
        """Test gradient flow through model"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        
        x = torch.randn(4, 2048, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_parameter_count(self):
        """Test model has reasonable number of parameters"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be between 1M and 50M parameters
        assert 1_000_000 < total_params < 50_000_000
    
    def test_save_load(self):
        """Test model save and load"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        
        # Save
        torch.save(model.state_dict(), 'test_model.pth')
        
        # Load
        model2 = CRNNClassifier(input_dim=2048, num_classes=5)
        model2.load_state_dict(torch.load('test_model.pth'))
        
        # Compare outputs
        x = torch.randn(4, 2048)
        model.eval()
        model2.eval()
        
        output1 = model(x)
        output2 = model2(x)
        
        assert torch.allclose(output1, output2, atol=1e-5)
        
        # Cleanup
        import os
        os.remove('test_model.pth')


class TestModelEdgeCases:
    """Test edge cases"""
    
    def test_single_sample(self):
        """Test with single sample"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        
        x = torch.randn(1, 2048)
        output = model(x)
        
        assert output.shape == (1, 5)
    
    def test_large_batch(self):
        """Test with large batch"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        
        x = torch.randn(128, 2048)
        output = model(x)
        
        assert output.shape == (128, 5)
    
    def test_zero_input(self):
        """Test with zero input"""
        model = CRNNClassifier(input_dim=2048, num_classes=5)
        model.eval()
        
        x = torch.zeros(4, 2048)
        output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
