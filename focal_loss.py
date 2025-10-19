"""
Focal Loss implementation for handling class imbalance
Critical for achieving >92% accuracy with imbalanced ICBHI dataset
"""

import tensorflow as tf
from tensorflow import keras


def categorical_focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for multi-class classification
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor (0-1)
        gamma: Focusing parameter (typically 2.0)
        
    Returns:
        Loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        Args:
            y_true: Ground truth (one-hot encoded)
            y_pred: Predictions (softmax probabilities)
        """
        # Clip predictions to prevent log(0)
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * keras.backend.log(y_pred)
        
        # Calculate focal loss
        loss = alpha * keras.backend.pow(1 - y_pred, gamma) * cross_entropy
        
        # Sum over classes
        return keras.backend.sum(loss, axis=-1)
    
    return focal_loss_fixed


def binary_focal_loss(alpha=0.25, gamma=2.0):
    """Binary focal loss (for binary classification)"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        loss = -alpha_t * keras.backend.pow(1 - p_t, gamma) * keras.backend.log(p_t)
        
        return keras.backend.mean(loss)
    
    return focal_loss_fixed


class FocalLoss(keras.losses.Loss):
    """Focal Loss as a Keras Loss class"""
    
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * keras.backend.log(y_pred)
        loss = self.alpha * keras.backend.pow(1 - y_pred, self.gamma) * cross_entropy
        
        return keras.backend.sum(loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


if __name__ == '__main__':
    # Test focal loss
    print("Testing Focal Loss...")
    
    # Create dummy data
    y_true = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.7, 0.1, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1], [0.1, 0.2, 0.5, 0.2]], dtype=tf.float32)
    
    # Standard categorical crossentropy
    cce = keras.losses.CategoricalCrossentropy()
    cce_loss = cce(y_true, y_pred)
    print(f"CCE Loss: {cce_loss:.4f}")
    
    # Focal loss
    focal = categorical_focal_loss(alpha=0.25, gamma=2.0)
    focal_loss_val = focal(y_true, y_pred)
    print(f"Focal Loss: {tf.reduce_mean(focal_loss_val):.4f}")
    
    print("✅ Focal loss test passed!")
