"""
Lightweight CNN for real-time laptop inference (<50ms)
Optimized for 3-second clips at 22.05kHz
Target: 85-88% accuracy, <2M parameters
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional


def create_lightweight_cnn(input_shape: Tuple[int, int, int] = (128, 96, 1),
                          num_classes: int = 4,
                          dropout_rate: float = 0.5,
                          l2_reg: float = 1e-4) -> Model:
    """
    Lightweight CNN optimized for CPU inference
    
    Architecture:
    - 4 Conv blocks with increasing filters
    - Global Average Pooling
    - Minimal fully connected layers
    - Batch Normalization for stability
    
    Args:
        input_shape: (time_steps, n_mels, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        l2_reg: L2 regularization factor
        
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape, name='spectrogram_input')
    
    x = inputs
    
    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding='same', 
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.6)(x)
    
    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    
    # Block 4: 256 filters
    x = layers.Conv2D(256, (3, 3), padding='same',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Global pooling (much faster than Flatten)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LightweightCNN')
    
    return model


def create_efficient_cnn(input_shape: Tuple[int, int, int] = (128, 96, 1),
                        num_classes: int = 4,
                        dropout_rate: float = 0.5) -> Model:
    """
    Even more efficient CNN using depthwise separable convolutions
    Similar to MobileNet architecture
    
    Target: <1M parameters, <30ms inference
    """
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    
    # Initial conv
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Depthwise separable blocks
    def depthwise_block(x, filters, strides=1):
        # Depthwise
        x = layers.DepthwiseConv2D((3, 3), strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Pointwise
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    x = depthwise_block(x, 64, strides=2)
    x = layers.Dropout(dropout_rate * 0.3)(x)
    
    x = depthwise_block(x, 128, strides=2)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    x = depthwise_block(x, 256)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    
    # Global pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='EfficientCNN')
    
    return model


def create_resnet_small(input_shape: Tuple[int, int, int] = (128, 96, 1),
                       num_classes: int = 4,
                       dropout_rate: float = 0.5) -> Model:
    """
    Small ResNet-style model with residual connections
    Better gradient flow for deeper training
    """
    inputs = keras.Input(shape=input_shape)
    
    def residual_block(x, filters, strides=1):
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if strides > 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add and activate
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    # Initial conv
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks
    x = residual_block(x, 64, strides=2)
    x = layers.Dropout(dropout_rate * 0.3)(x)
    
    x = residual_block(x, 128, strides=2)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    x = residual_block(x, 256, strides=2)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    
    # Global pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ResNetSmall')
    
    return model


def create_model_with_augmentation(base_model: Model,
                                  use_specaugment: bool = True) -> Model:
    """
    Wrap base model with augmentation layers
    
    Args:
        base_model: Base CNN model
        use_specaugment: Enable SpecAugment
        
    Returns:
        Model with augmentation
    """
    from augmentation_v2 import AudioAugmentation
    
    inputs = keras.Input(shape=base_model.input_shape[1:])
    
    # Augmentation layer (only active during training)
    if use_specaugment:
        x = AudioAugmentation(spec_augment=True)(inputs)
    else:
        x = inputs
    
    # Base model
    outputs = base_model(x)
    
    model = Model(inputs=inputs, outputs=outputs, 
                 name=f'{base_model.name}_with_aug')
    
    return model


def compile_model(model: Model,
                 learning_rate: float = 1e-3,
                 use_focal_loss: bool = True,
                 class_weights: Optional[np.ndarray] = None,
                 label_smoothing: float = 0.1) -> Model:
    """
    Compile model with appropriate loss and metrics
    
    Args:
        model: Keras model
        learning_rate: Initial learning rate
        use_focal_loss: Use focal loss instead of categorical crossentropy
        class_weights: Class weights for imbalanced data
        label_smoothing: Label smoothing factor
        
    Returns:
        Compiled model
    """
    
    # Optimizer with weight decay (AdamW)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Loss function
    if use_focal_loss:
        from focal_loss import categorical_focal_loss
        loss = categorical_focal_loss(alpha=0.25, gamma=2.0)
    else:
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    
    # Metrics
    metrics = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_model_summary(model: Model) -> dict:
    """Get model statistics"""
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    return {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }


def benchmark_inference(model: Model, 
                       input_shape: Tuple[int, int, int],
                       num_iterations: int = 100) -> dict:
    """
    Benchmark model inference speed
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    # Warmup
    dummy_input = tf.random.normal([1] + list(input_shape))
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99))
    }


def main():
    """Test model creation and benchmarking"""
    print("="*80)
    print("LIGHTWEIGHT CNN MODELS")
    print("="*80)
    
    input_shape = (128, 96, 1)
    num_classes = 4
    
    # Create models
    models = {
        'Lightweight': create_lightweight_cnn(input_shape, num_classes),
        'Efficient': create_efficient_cnn(input_shape, num_classes),
        'ResNetSmall': create_resnet_small(input_shape, num_classes)
    }
    
    for name, model in models.items():
        print(f"\n{name} Model:")
        print("-"*80)
        
        # Summary
        summary = get_model_summary(model)
        print(f"Parameters: {summary['total_params']:,}")
        print(f"Size: {summary['size_mb']:.2f} MB")
        print(f"Layers: {summary['layers']}")
        
        # Benchmark
        print("\nBenchmarking inference speed...")
        benchmark = benchmark_inference(model, input_shape, num_iterations=50)
        print(f"Mean: {benchmark['mean_ms']:.2f}ms")
        print(f"Std: {benchmark['std_ms']:.2f}ms")
        print(f"P95: {benchmark['p95_ms']:.2f}ms")
        print(f"P99: {benchmark['p99_ms']:.2f}ms")
        
        # Check if meets requirements
        if benchmark['p95_ms'] < 50:
            print("✅ Meets <50ms requirement")
        else:
            print(f"⚠️ Exceeds 50ms target ({benchmark['p95_ms']:.2f}ms)")
    
    print("\n" + "="*80)
    print("Model comparison complete!")


if __name__ == '__main__':
    main()
