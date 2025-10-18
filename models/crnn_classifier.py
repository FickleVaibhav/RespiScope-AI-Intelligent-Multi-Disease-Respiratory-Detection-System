"""
CRNN (CNN + RNN) Classifier for Respiratory Disease Detection
Combines CNN for spatial features and RNN for temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CRNNClassifier(nn.Module):
    """
    CRNN architecture for multi-class respiratory disease classification
    Uses CNN for feature extraction and Bi-LSTM for temporal modeling
    """
    
    def __init__(self,
                 input_dim: int = 2048,
                 cnn_channels: List[int] = [128, 256, 512],
                 rnn_hidden: int = 256,
                 rnn_layers: int = 2,
                 num_classes: int = 5,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize CRNN Classifier
        
        Args:
            input_dim: Dimension of input features (PANN embedding size)
            cnn_channels: List of CNN channel sizes
            rnn_hidden: Hidden size of RNN
            rnn_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional RNN
        """
        super(CRNNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.rnn_hidden = rnn_hidden
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, cnn_channels[0])
        self.bn_input = nn.BatchNorm1d(cnn_channels[0])
        
        # CNN layers for spatial feature extraction
        self.cnn_layers = nn.ModuleList()
        for i in range(len(cnn_channels) - 1):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        cnn_channels[i],
                        cnn_channels[i + 1],
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm1d(cnn_channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.MaxPool1d(kernel_size=2)
                )
            )
        
        # Bi-directional LSTM for temporal modeling
        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        rnn_output_size = rnn_hidden * 2 if bidirectional else rnn_hidden
        self.attention = AttentionLayer(rnn_output_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(rnn_output_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(256, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            return_features: If True, return intermediate features
            
        Returns:
            Class logits and optionally features
        """
        batch_size = x.size(0)
        
        # Handle 2D input (batch_size, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Input projection
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        x = self.input_proj(x.transpose(1, 2))
        x = self.bn_input(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        
        # CNN feature extraction
        cnn_features = x.transpose(1, 2)  # (batch, channels, seq_len)
        for cnn_layer in self.cnn_layers:
            cnn_features = cnn_layer(cnn_features)
        
        # Prepare for RNN
        rnn_input = cnn_features.transpose(1, 2)  # (batch, seq_len, channels)
        
        # RNN temporal modeling
        rnn_output, (hidden, cell) = self.rnn(rnn_input)
        
        # Attention pooling
        attended_output = self.attention(rnn_output)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(attended_output)))
        x = self.dropout1(x)
        
        features = F.relu(self.bn_fc2(self.fc2(x)))
        features = self.dropout2(features)
        
        # Output
        logits = self.fc_out(features)
        
        if return_features:
            return logits, features
        return logits


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence pooling"""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, rnn_output):
        """
        Apply attention to RNN output
        
        Args:
            rnn_output: RNN output (batch, seq_len, hidden_size)
            
        Returns:
            Attended output (batch, hidden_size)
        """
        # Compute attention scores
        attention_scores = self.attention(rnn_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        attended = torch.sum(rnn_output * attention_weights, dim=1)
        
        return attended


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for respiratory disease detection
    Alternative to CRNN with self-attention mechanism
    """
    
    def __init__(self,
                 input_dim: int = 2048,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 num_classes: int = 5,
                 dropout: float = 0.1,
                 max_seq_len: int = 1000):
        """
        Initialize Transformer Classifier
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            num_classes: Number of output classes
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(TransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            return_features: If True, return intermediate features
            
        Returns:
            Class logits and optionally features
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)
        
        if return_features:
            return logits, x
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RespiScopeModel(nn.Module):
    """
    Complete RespiScope model combining PANN and classifier
    """
    
    def __init__(self,
                 pann_model,
                 classifier_type: str = 'crnn',
                 classifier_config: dict = None,
                 freeze_pann: bool = False):
        """
        Initialize RespiScope model
        
        Args:
            pann_model: PANN feature extractor
            classifier_type: Type of classifier ('crnn' or 'transformer')
            classifier_config: Configuration for classifier
            freeze_pann: Whether to freeze PANN weights
        """
        super(RespiScopeModel, self).__init__()
        
        self.pann_model = pann_model
        self.classifier_type = classifier_type
        
        # Freeze PANN if requested
        if freeze_pann:
            for param in self.pann_model.parameters():
                param.requires_grad = False
        
        # Initialize classifier
        if classifier_config is None:
            classifier_config = {}
        
        if classifier_type == 'crnn':
            self.classifier = CRNNClassifier(**classifier_config)
        elif classifier_type == 'transformer':
            self.classifier = TransformerClassifier(**classifier_config)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def forward(self, audio, return_features=False):
        """
        Forward pass through complete model
        
        Args:
            audio: Input audio tensor
            return_features: Whether to return features
            
        Returns:
            Logits and optionally features
        """
        # Extract PANN embeddings
        with torch.set_grad_enabled(not self.training or self.pann_model.training):
            embeddings = self.pann_model.extract_embedding(audio, batch_process=True)
            embeddings = torch.from_numpy(embeddings).to(audio.device)
        
        # Classify
        output = self.classifier(embeddings, return_features=return_features)
        
        return output


def create_model(model_type: str = 'crnn',
                 num_classes: int = 5,
                 pann_config: dict = None,
                 classifier_config: dict = None,
                 device: str = 'cuda') -> RespiScopeModel:
    """
    Factory function to create RespiScope model
    
    Args:
        model_type: Type of classifier ('crnn' or 'transformer')
        num_classes: Number of output classes
        pann_config: Configuration for PANN
        classifier_config: Configuration for classifier
        device: Device to use
        
    Returns:
        RespiScopeModel instance
    """
    from .pann_embeddings import load_pann_extractor
    
    # Default configs
    if pann_config is None:
        pann_config = {'model_name': 'Cnn14', 'sample_rate': 16000}
    
    if classifier_config is None:
        if model_type == 'crnn':
            classifier_config = {
                'input_dim': 2048,
                'cnn_channels': [128, 256, 512],
                'rnn_hidden': 256,
                'rnn_layers': 2,
                'num_classes': num_classes,
                'dropout': 0.3
            }
        elif model_type == 'transformer':
            classifier_config = {
                'input_dim': 2048,
                'd_model': 512,
                'nhead': 8,
                'num_layers': 6,
                'num_classes': num_classes,
                'dropout': 0.1
            }
    
    # Create PANN extractor
    pann_model = load_pann_extractor(device=device, **pann_config)
    
    # Create complete model
    model = RespiScopeModel(
        pann_model=pann_model,
        classifier_type=model_type,
        classifier_config=classifier_config,
        freeze_pann=False
    )
    
    return model.to(device)


if __name__ == "__main__":
    # Test models
    print("Testing CRNN Classifier...")
    
    # Create dummy input (PANN embedding)
    batch_size = 4
    input_dim = 2048
    dummy_input = torch.randn(batch_size, input_dim)
    
    # Test CRNN
    crnn_model = CRNNClassifier(
        input_dim=input_dim,
        num_classes=5
    )
    crnn_output = crnn_model(dummy_input)
    print(f"CRNN Input shape: {dummy_input.shape}")
    print(f"CRNN Output shape: {crnn_output.shape}")
    
    # Test Transformer
    print("\nTesting Transformer Classifier...")
    transformer_model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=5
    )
    transformer_output = transformer_model(dummy_input)
    print(f"Transformer Input shape: {dummy_input.shape}")
    print(f"Transformer Output shape: {transformer_output.shape}")
    
    # Count parameters
    crnn_params = sum(p.numel() for p in crnn_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    
    print(f"\nCRNN Parameters: {crnn_params:,}")
    print(f"Transformer Parameters: {transformer_params:,}")
    
    print("\nModel test successful!")
