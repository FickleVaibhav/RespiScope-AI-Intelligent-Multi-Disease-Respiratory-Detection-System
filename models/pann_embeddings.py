"""
PANN (Pretrained Audio Neural Networks) Feature Extraction
Uses pretrained CNN14 model from PANNs for audio embedding extraction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
import torchaudio
import warnings
warnings.filterwarnings('ignore')


class Cnn14(nn.Module):
    """
    CNN14 architecture from PANNs
    Pretrained on AudioSet for audio pattern recognition
    """
    
    def __init__(self, 
                 sample_rate: int = 32000,
                 window_size: int = 1024,
                 hop_size: int = 320,
                 mel_bins: int = 64,
                 fmin: int = 50,
                 fmax: int = 14000,
                 classes_num: int = 527):
        """
        Initialize CNN14 model
        
        Args:
            sample_rate: Audio sample rate
            window_size: FFT window size
            hop_size: Hop size between frames
            mel_bins: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            classes_num: Number of output classes (AudioSet has 527)
        """
        super(Cnn14, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        # Spectrogram extractor
        self.spectrogram_extractor = torchaudio.transforms.Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window_fn=torch.hann_window,
            center=center,
            pad_mode=pad_mode,
            power=2.0,
        )
        
        # Logmel feature extractor
        self.logmel_extractor = torchaudio.transforms.MelScale(
            n_mels=mel_bins,
            sample_rate=sample_rate,
            f_min=fmin,
            f_max=fmax,
            n_stft=window_size // 2 + 1,
        )
        
        self.bn0 = nn.BatchNorm2d(64)
        
        # CNN blocks
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()
    
    def init_weight(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input_audio, return_embedding=True):
        """
        Forward pass
        
        Args:
            input_audio: Input audio tensor (batch_size, audio_length)
            return_embedding: If True, return embedding instead of classification
            
        Returns:
            embeddings or classification logits
        """
        # Compute spectrogram
        x = self.spectrogram_extractor(input_audio)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        
        # Log transform
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = x.unsqueeze(1)  # Add channel dimension
        
        # CNN blocks with pooling
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = nn.functional.relu_(self.fc1(x))
        embedding = nn.functional.dropout(x, p=0.5, training=self.training)
        
        if return_embedding:
            return embedding
        else:
            clipwise_output = torch.sigmoid(self.fc_audioset(x))
            return clipwise_output


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        """Forward pass with pooling"""
        x = nn.functional.relu_(self.bn1(self.conv1(x)))
        x = nn.functional.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'max':
            x = nn.functional.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = nn.functional.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = nn.functional.avg_pool2d(x, kernel_size=pool_size)
            x2 = nn.functional.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception(f'Incorrect pool_type: {pool_type}')
        
        return x


class PANNFeatureExtractor(nn.Module):
    """
    Wrapper for PANN feature extraction
    Handles audio preprocessing and embedding extraction
    """
    
    def __init__(self,
                 model_name: str = 'Cnn14',
                 sample_rate: int = 16000,
                 pretrained: bool = True,
                 device: str = 'cuda'):
        """
        Initialize PANN feature extractor
        
        Args:
            model_name: PANN model name (Cnn14, Cnn10, Cnn6)
            sample_rate: Input audio sample rate
            pretrained: Whether to load pretrained weights
            device: Device to run model on
        """
        super(PANNFeatureExtractor, self).__init__()
        
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        
        # Initialize model
        if model_name == 'Cnn14':
            self.model = Cnn14(sample_rate=32000)  # PANN uses 32kHz
        else:
            raise ValueError(f"Model {model_name} not implemented")
        
        # Load pretrained weights if requested
        if pretrained:
            self.load_pretrained_weights()
        
        self.model.to(device)
        self.model.eval()
        
        # Resampler if needed (PANN expects 32kHz)
        if sample_rate != 32000:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=32000
            ).to(device)
        else:
            self.resampler = None
    
    def load_pretrained_weights(self):
        """Load pretrained weights from PANNs"""
        try:
            import wget
            import os
            
            # Download pretrained weights
            checkpoint_path = f"checkpoints/{self.model_name}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            
            if not os.path.exists(checkpoint_path):
                url = f"https://zenodo.org/record/3987831/files/{self.model_name}_mAP%3D0.431.pth"
                print(f"Downloading pretrained weights from {url}")
                wget.download(url, checkpoint_path)
                print("\nDownload complete!")
            
            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'], strict=False)
            print(f"Loaded pretrained weights for {self.model_name}")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized weights")
    
    @torch.no_grad()
    def extract_embedding(self, 
                         audio: Union[np.ndarray, torch.Tensor],
                         batch_process: bool = False) -> np.ndarray:
        """
        Extract embedding from audio
        
        Args:
            audio: Input audio (numpy array or torch tensor)
            batch_process: Whether input is batched
            
        Returns:
            Audio embedding (2048-dimensional)
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Add batch dimension if needed
        if not batch_process and audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        # Move to device
        audio = audio.to(self.device)
        
        # Resample if needed
        if self.resampler is not None:
            audio = self.resampler(audio)
        
        # Extract embedding
        embedding = self.model(audio, return_embedding=True)
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()
        
        return embedding
    
    def __call__(self, audio):
        """Make class callable"""
        return self.extract_embedding(audio)


def load_pann_extractor(model_name: str = 'Cnn14',
                        sample_rate: int = 16000,
                        device: str = 'cuda') -> PANNFeatureExtractor:
    """
    Convenience function to load PANN feature extractor
    
    Args:
        model_name: PANN model name
        sample_rate: Audio sample rate
        device: Device to use
        
    Returns:
        PANNFeatureExtractor instance
    """
    return PANNFeatureExtractor(
        model_name=model_name,
        sample_rate=sample_rate,
        pretrained=True,
        device=device
    )


if __name__ == "__main__":
    # Test PANN feature extraction
    print("Testing PANN Feature Extractor...")
    
    # Create dummy audio (16kHz, 5 seconds)
    sample_rate = 16000
    duration = 5
    dummy_audio = np.random.randn(sample_rate * duration).astype(np.float32)
    
    # Initialize extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = load_pann_extractor(
        model_name='Cnn14',
        sample_rate=sample_rate,
        device=device
    )
    
    print(f"Using device: {device}")
    
    # Extract embedding
    embedding = extractor.extract_embedding(dummy_audio)
    
    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Embedding statistics:")
    print(f"  Mean: {embedding.mean():.4f}")
    print(f"  Std: {embedding.std():.4f}")
    print(f"  Min: {embedding.min():.4f}")
    print(f"  Max: {embedding.max():.4f}")
    
    print("\nPANN feature extractor test successful!")
