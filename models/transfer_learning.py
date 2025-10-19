"""
Transfer learning with YAMNet for >92% accuracy
YAMNet is pretrained on AudioSet and optimized for audio classification
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_yamnet_transfer_model(num_classes: int = 4,
                                 trainable_layers: int = 3,
                                 dropout_rate: float = 0.5) -> keras.Model:
    """
    Create transfer learning model using YAMNet
    
    Args:
        num_classes: Number of output classes
        trainable_layers: Number of top layers to fine-tune
        dropout_rate: Dropout rate
        
    Returns:
        Keras Model
    """
    # YAMNet expects waveform input at 16kHz
    # We'll use YAMNet embeddings (1024-D)
    
    yamnet_model_url = 'https://tfhub.dev/google/yamnet/1'
    
    # Load YAMNet
    yamnet_model = hub.KerasLayer(yamnet_model_url, trainable=False)
    
    # Input is waveform
    inputs = keras.Input(shape=(None,), name='audio_input')
    
    # Extract embeddings
    _, embeddings, _ = yamnet_model(inputs)
    
    # Pool embeddings (YAMNet produces variable length)
    x = layers.GlobalAveragePooling1D()(embeddings)
    
    # Classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='YAMNet_Transfer')
    
    return model


def create_yamnet_spec_transfer(input_shape: tuple = (128, 96, 1),
                                num_classes: int = 4,
                                dropout_rate: float = 0.5) -> keras.Model:
    """
    Transfer learning using spectrogram-based approach
    More compatible with our preprocessing pipeline
    """
    
    # Use EfficientNet as base (pretrained on ImageNet)
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape if input_shape[2] == 3 else (*input_shape[:2], 3),
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    
    # Convert single channel to 3 channels if needed
    if input_shape[2] == 1:
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
    else:
        x = inputs
    
    # Base model
    x = base_model(x)
    
    # Classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNet_Transfer')
    
    return model


def unfreeze_model(model: keras.Model, num_layers: int = 20):
    """
    Unfreeze top layers for fine-tuning
    
    Args:
        model: Keras model
        num_layers: Number of layers to unfreeze from top
    """
    # Find the base model
    base_model = None
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower() or 'yamnet' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find base model to unfreeze")
        return model
    
    # Unfreeze top layers
    base_model.trainable = True
    
    # Freeze all but last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Unfroze {trainable_count} layers in base model")
    
    return model


def create_vggish_transfer(input_shape: tuple = (128, 96, 1),
                           num_classes: int = 4,
                           dropout_rate: float = 0.5) -> keras.Model:
    """
    VGGish-inspired model for audio
    Similar architecture to VGG but optimized for spectrograms
    """
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    
    # VGG-style blocks
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.3)(x)
    
    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.4)(x)
    
    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    outputs
