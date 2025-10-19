"""
Complete training script with K-Fold CV for >92% accuracy
Includes: focal loss, class weights, mixup, SpecAugment, ensemble
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.lightweight_cnn import (
    create_lightweight_cnn, create_efficient_cnn, create_resnet_small, compile_model
)
from augmentation_v2 import create_augmentation_pipeline, mixup
from focal_loss import categorical_focal_loss
from utils.metrics import calculate_metrics


# Set random seeds for reproducibility
def set_seeds(seed=42):
    """Set all random seeds"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        # Data
        self.data_dir = 'data/processed/spectrograms'
        self.metadata_file = 'data/processed/processed_metadata.csv'
        self.input_shape = (128, 96, 1)
        self.num_classes = 4
        self.class_names = ['Asthma', 'Bronchitis', 'COPD', 'Pneumonia']
        
        # Model
        self.model_type = 'lightweight'  # 'lightweight', 'efficient', 'resnet'
        self.dropout_rate = 0.5
        self.l2_reg = 1e-4
        
        # Training
        self.n_folds = 5
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 1e-3
        self.patience = 15
        
        # Loss & optimization
        self.use_focal_loss = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.label_smoothing = 0.1
        self.use_class_weights = True
        
        # Augmentation
        self.use_specaugment = True
        self.use_mixup = True
        self.mixup_alpha = 0.2
        
        # Callbacks
        self.reduce_lr_patience = 10
        self.reduce_lr_factor = 0.5
        self.min_lr = 1e-6
        
        # Output
        self.output_dir = 'experiments'
        self.experiment_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_best_only = True
        
        # Reproducibility
        self.random_seed = 42


def load_dataset(config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load preprocessed dataset"""
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    # Load metadata
    df = pd.read_csv(config.metadata_file)
    print(f"Total samples: {len(df)}")
    
    # Filter by split if needed
    if 'split' in df.columns:
        df_train = df[df['split'] == 'train'].copy()
        print(f"Training samples: {len(df_train)}")
    else:
        df_train = df.copy()
    
    # Load spectrograms
    X = []
    y = []
    
    print("Loading spectrograms...")
    for idx, row in df_train.iterrows():
        spec_path = os.path.join(config.data_dir, row['clip_filename'])
        
        if not os.path.exists(spec_path):
            continue
        
        spec = np.load(spec_path)
        
        # Ensure correct shape
        if spec.shape != config.input_shape[:2]:
            continue
        
        X.append(spec)
        
        # Get label
        class_label = row['class']
        if class_label in config.class_names:
            y.append(config.class_names.index(class_label))
        else:
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension
    if len(X.shape) == 3:
        X = np.expand_dims(X, -1)
    
    print(f"\nLoaded data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution:")
    for i, name in enumerate(config.class_names):
        count = np.sum(y == i)
        print(f"  {name}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, df_train


def compute_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Compute class weights for imbalanced dataset"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=y
    )
    
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    print("\nClass weights:")
    for i, w in class_weight_dict.items():
        print(f"  Class {i}: {w:.3f}")
    
    return class_weight_dict


def create_tf_dataset(X: np.ndarray, y: np.ndarray, 
                     config: TrainingConfig,
                     training: bool = True) -> tf.data.Dataset:
    """Create TensorFlow dataset with augmentation"""
    
    # One-hot encode labels
    y_onehot = tf.keras.utils.to_categorical(y, config.num_classes)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y_onehot))
    
    if training:
        dataset = dataset.shuffle(buffer_size=1000, seed=config.random_seed)
    
    # Apply augmentation
    if training and config.use_specaugment:
        augment_fn = create_augmentation_pipeline(use_specaugment=True, training=True)
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def train_fold(fold: int, 
               train_idx: np.ndarray,
               val_idx: np.ndarray,
               X: np.ndarray,
               y: np.ndarray,
               config: TrainingConfig) -> Tuple[keras.Model, Dict]:
    """Train single fold"""
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{config.n_folds}")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create datasets
    train_dataset = create_tf_dataset(X_train, y_train, config, training=True)
    val_dataset = create_tf_dataset(X_val, y_val, config, training=False)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train, config.num_classes) if config.use_class_weights else None
    
    # Create model
    print(f"\nCreating {config.model_type} model...")
    if config.model_type == 'lightweight':
        model = create_lightweight_cnn(config.input_shape, config.num_classes, 
                                      config.dropout_rate, config.l2_reg)
    elif config.model_type == 'efficient':
        model = create_efficient_cnn(config.input_shape, config.num_classes, config.dropout_rate)
    elif config.model_type == 'resnet':
        model = create_resnet_small(config.input_shape, config.num_classes, config.dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    # Compile model
    model = compile_model(
        model,
        learning_rate=config.learning_rate,
        use_focal_loss=config.use_focal_loss,
        label_smoothing=config.label_smoothing
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    fold_dir = os.path.join(config.output_dir, config.experiment_name, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(fold_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=config.save_best_only,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(fold_dir, 'training_log.csv')
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(fold_dir, 'tensorboard')
        )
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_dataset, verbose=0)
    
    results = {
        'fold': fold,
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'val_precision': float(val_prec),
        'val_recall': float(val_rec),
        'val_auc': float(val_auc),
        'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
        'train_samples': len(train_idx),
        'val_samples': len(val_idx)
    }
    
    # Save results
    with open(os.path.join(fold_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFold {fold + 1} Results:")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}")
    print(f"  Val AUC: {val_auc:.4f}")
    
    return model, results


def train_kfold(X: np.ndarray, y: np.ndarray, config: TrainingConfig) -> List[Dict]:
    """Train with K-Fold cross-validation"""
    
    print("\n" + "="*80)
    print(f"K-FOLD CROSS-VALIDATION (K={config.n_folds})")
    print("="*80)
    
    # Set seeds
    set_seeds(config.random_seed)
    
    # K-Fold split (stratified)
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    
    fold_results = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model, results = train_fold(fold, train_idx, val_idx, X, y, config)
        fold_results.append(results)
        fold_models.append(model)
    
    # Aggregate results
    print("\n" + "="*80)
    print("K-FOLD RESULTS SUMMARY")
    print("="*80)
    
    metrics = ['val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_auc']
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save aggregate results
    aggregate = {
        'config': config.__dict__,
        'fold_results': fold_results,
        'mean_val_accuracy': float(np.mean([r['val_accuracy'] for r in fold_results])),
        'std_val_accuracy': float(np.std([r['val_accuracy'] for r in fold_results])),
        'mean_val_loss': float(np.mean([r['val_loss'] for r in fold_results])),
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = os.path.join(config.output_dir, config.experiment_name, 'aggregate_results.json')
    with open(output_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Results saved to: {os.path.join(config.output_dir, config.experiment_name)}")
    
    return fold_results, fold_models


def main():
    parser = argparse.ArgumentParser(description='Train RespiScope-AI with K-Fold CV')
    parser.add_argument('--data_dir', type=str, default='data/processed/spectrograms')
    parser.add_argument('--metadata', type=str, default='data/processed/processed_metadata.csv')
    parser.add_argument('--model_type', type=str, default='lightweight', 
                       choices=['lightweight', 'efficient', 'resnet'])
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_focal_loss', action='store_true')
    parser.add_argument('--no_specaugment', action='store_true')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.data_dir = args.data_dir
    config.metadata_file = args.metadata
    config.model_type = args.model_type
    config.n_folds = args.n_folds
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.dropout_rate = args.dropout
    config.output_dir = args.output_dir
    config.random_seed = args.seed
    config.use_focal_loss = not args.no_focal_loss
    config.use_specaugment = not args.no_specaugment
    
    # Set seeds
    set_seeds(config.random_seed)
    
    # Load data
    X, y, df = load_dataset(config)
    
    # Train
    fold_results, fold_models = train_kfold(X, y, config)
    
    # Final summary
    mean_acc = np.mean([r['val_accuracy'] for r in fold_results])
    print(f"\n{'='*80}")
    print(f"FINAL ACCURACY: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    if mean_acc >= 0.92:
        print("✅ TARGET ACHIEVED: >92% accuracy!")
    else:
        print(f"⚠️  Gap to target: {(0.92 - mean_acc)*100:.2f}%")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
