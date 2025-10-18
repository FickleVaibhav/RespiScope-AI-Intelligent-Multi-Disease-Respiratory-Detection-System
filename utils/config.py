"""
Configuration file for RespiScope-AI
Centralized configuration management for all modules
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AudioConfig:
    """Audio preprocessing configuration"""
    sample_rate: int = 16000  # Target sampling rate
    duration: float = 10.0  # Maximum audio duration in seconds
    hop_length: int = 512
    n_mels: int = 128
    n_fft: int = 2048
    fmin: int = 50  # Minimum frequency
    fmax: int = 8000  # Maximum frequency
    top_db: int = 30  # Threshold for silence removal
    

@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    enabled: bool = True
    pitch_shift_prob: float = 0.3
    pitch_shift_range: tuple = (-2, 2)  # Semitones
    time_stretch_prob: float = 0.3
    time_stretch_range: tuple = (0.9, 1.1)
    noise_injection_prob: float = 0.3
    noise_snr_range: tuple = (10, 30)  # SNR in dB
    time_mask_prob: float = 0.2
    freq_mask_prob: float = 0.2


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # PANN Configuration
    pann_model: str = "Cnn14"  # Options: Cnn14, Cnn10, Cnn6
    pann_embedding_dim: int = 2048
    
    # CRNN Configuration
    crnn_cnn_channels: List[int] = None
    crnn_rnn_hidden: int = 256
    crnn_rnn_layers: int = 2
    crnn_dropout: float = 0.3
    
    # Transformer Configuration
    transformer_d_model: int = 512
    transformer_nhead: int = 8
    transformer_num_layers: int = 6
    transformer_dim_feedforward: int = 2048
    transformer_dropout: float = 0.1
    
    # Output
    num_classes: int = 5
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.crnn_cnn_channels is None:
            self.crnn_cnn_channels = [128, 256, 512]
        if self.class_names is None:
            self.class_names = ['Asthma', 'Bronchitis', 'COPD', 'Pneumonia', 'Healthy']


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # Options: cosine, step, plateau
    scheduler_patience: int = 10
    early_stopping_patience: int = 15
    gradient_clip: float = 1.0
    
    # Loss function
    loss_fn: str = "cross_entropy"  # Options: cross_entropy, focal_loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Class weights (optional, for imbalanced datasets)
    use_class_weights: bool = True
    
    # Mixed precision training
    use_amp: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 5
    save_top_k: int = 3


@dataclass
class DataConfig:
    """Dataset configuration"""
    data_root: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Supported datasets
    datasets: List[str] = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ['icbhi', 'coswara']


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_path: str = "models/checkpoints/best_model.pth"
    device: str = "cuda"  # cuda or cpu
    batch_size: int = 1
    use_tta: bool = False  # Test-time augmentation
    confidence_threshold: float = 0.5


@dataclass
class WebAppConfig:
    """Web application configuration"""
    gradio_port: int = 7860
    streamlit_port: int = 8501
    max_file_size_mb: int = 50
    allowed_formats: List[str] = None
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']


@dataclass
class Config:
    """Main configuration class"""
    audio: AudioConfig = None
    augmentation: AugmentationConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    inference: InferenceConfig = None
    webapp: WebAppConfig = None
    
    # Hardware settings
    num_workers: int = 4
    device: str = "cuda"
    
    # Logging
    log_dir: str = "logs"
    experiment_name: str = "respiscope_experiment"
    
    # Paths
    checkpoint_dir: str = "models/checkpoints"
    results_dir: str = "results"
    
    def __post_init__(self):
        if self.audio is None:
            self.audio = AudioConfig()
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.webapp is None:
            self.webapp = WebAppConfig()
        
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


def get_config() -> Config:
    """Get default configuration"""
    return Config()


def save_config(config: Config, path: str):
    """Save configuration to YAML file"""
    import yaml
    from dataclasses import asdict
    
    config_dict = asdict(config)
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Configuration saved to {path}")


def load_config(path: str) -> Config:
    """Load configuration from YAML file"""
    import yaml
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Reconstruct Config object from dict
    config = Config()
    config.audio = AudioConfig(**config_dict.get('audio', {}))
    config.augmentation = AugmentationConfig(**config_dict.get('augmentation', {}))
    config.model = ModelConfig(**config_dict.get('model', {}))
    config.training = TrainingConfig(**config_dict.get('training', {}))
    config.data = DataConfig(**config_dict.get('data', {}))
    config.inference = InferenceConfig(**config_dict.get('inference', {}))
    config.webapp = WebAppConfig(**config_dict.get('webapp', {}))
    
    # Update top-level attributes
    for key, value in config_dict.items():
        if key not in ['audio', 'augmentation', 'model', 'training', 'data', 'inference', 'webapp']:
            setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print("Default Configuration:")
    print(f"Sample Rate: {config.audio.sample_rate}")
    print(f"Number of Classes: {config.model.num_classes}")
    print(f"Class Names: {config.model.class_names}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    
    # Save configuration
    save_config(config, "config.yaml")
    
    # Load configuration
    loaded_config = load_config("config.yaml")
    print("\nConfiguration loaded successfully!")
