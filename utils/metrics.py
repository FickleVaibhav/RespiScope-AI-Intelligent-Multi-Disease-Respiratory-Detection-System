"""
Evaluation metrics for RespiScope-AI
Provides comprehensive metrics for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from typing import List, Dict, Optional
import pandas as pd


def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_probs: np.ndarray,
                     class_names: List[str]) -> Dict:
    """
    Calculate comprehensive metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics['recall_per_class'] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics['f1_per_class'] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Weighted metrics
    metrics['weighted_precision'] = precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_recall'] = recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_f1'] = f1_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Macro metrics
    metrics['macro_precision'] = precision_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['macro_recall'] = recall_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['macro_f1'] = f1_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC (one-vs-rest)
    try:
        metrics['roc_auc_per_class'] = roc_auc_score(
            y_true, y_probs, multi_class='ovr', average=None
        )
        metrics['weighted_roc_auc'] = roc_auc_score(
            y_true, y_probs, multi_class='ovr', average='weighted'
        )
    except:
        metrics['roc_auc_per_class'] = None
        metrics['weighted_roc_auc'] = None
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         save_path: Optional[str] = None,
                         normalize: bool = True,
                         figsize: tuple = (10, 8)):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize
        figsize: Figure size
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(y_true: np.ndarray,
                   y_probs: np.ndarray,
                   class_names: List[str],
                   save_path: Optional[str] = None,
                   figsize: tuple = (12, 8)):
    """
    Plot ROC curves for all classes
    
    Args:
        y_true: True labels
        y_probs: Prediction probabilities
        class_names: List of class names
        save_path: Path to save plot
        figsize: Figure size
    """
    n_classes = len(class_names)
    
    # Binarize labels for one-vs-rest
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=figsize)
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr,
            label=f'{class_name} (AUC = {roc_auc:.3f})',
            linewidth=2
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - One-vs-Rest', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_metrics(metrics: Dict,
                          class_names: List[str],
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 6)):
    """
    Plot per-class precision, recall, and F1-score
    
    Args:
        metrics: Dictionary containing metrics
        class_names: List of class names
        save_path: Path to save plot
        figsize: Figure size
    """
    precision = metrics['precision_per_class']
    recall = metrics['recall_per_class']
    f1 = metrics['f1_per_class']
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=figsize)
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Per-Class Metrics', fontsize=14)
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history: Dict,
                         save_path: Optional[str] = None,
                         figsize: tuple = (14, 5)):
    """
    Plot training history
    
    Args:
        history: Dictionary with train/val losses and accuracies
        save_path: Path to save plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_losses'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot accuracy
    if 'train_accuracies' in history:
        axes[1].plot(history['train_accuracies'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracies'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_metrics_report(metrics: Dict,
                       class_names: List[str],
                       save_path: str):
    """
    Save comprehensive metrics report
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        save_path: Path to save report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RespiScope-AI Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Weighted Precision: {metrics['weighted_precision']:.4f}\n")
        f.write(f"Weighted Recall: {metrics['weighted_recall']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n")
        if metrics['weighted_roc_auc'] is not None:
            f.write(f"Weighted ROC-AUC: {metrics['weighted_roc_auc']:.4f}\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 80 + "\n")
        
        df = pd.DataFrame({
            'Class': class_names,
            'Precision': metrics['precision_per_class'],
            'Recall': metrics['recall_per_class'],
            'F1-Score': metrics['f1_per_class']
        })
        
        if metrics['roc_auc_per_class'] is not None:
            df['ROC-AUC'] = metrics['roc_auc_per_class']
        
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Classification report
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n")
        f.write(metrics['classification_report'])
        f.write("\n")
        
        # Confusion matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n")
        cm_df = pd.DataFrame(
            metrics['confusion_matrix'],
            index=[f"True_{name}" for name in class_names],
            columns=[f"Pred_{name}" for name in class_names]
        )
        f.write(cm_df.to_string())
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Metrics report saved to {save_path}")


def evaluate_model(model, test_loader, device, class_names, save_dir):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        class_names: List of class names
        save_dir: Directory to save results
    """
    import os
    import torch
    from tqdm import tqdm
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for audio, labels in tqdm(test_loader, desc='Evaluating'):
            audio = audio.to(device)
            labels = labels.to(device)
            
            outputs = model(audio)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(all_labels, all_preds, all_probs, class_names)
    
    # Save metrics report
    save_metrics_report(
        metrics,
        class_names,
        os.path.join(save_dir, 'evaluation_report.txt')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=os.path.join(save_dir, 'confusion_matrix.png'),
        normalize=True
    )
    
    # Plot ROC curves
    if metrics['roc_auc_per_class'] is not None:
        plot_roc_curves(
            all_labels,
            all_probs,
            class_names,
            save_path=os.path.join(save_dir, 'roc_curves.png')
        )
    
    # Plot per-class metrics
    plot_per_class_metrics(
        metrics,
        class_names,
        save_path=os.path.join(save_dir, 'per_class_metrics.png')
    )
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'True_Label': [class_names[i] for i in all_labels],
        'Predicted_Label': [class_names[i] for i in all_preds],
        **{f'Prob_{name}': all_probs[:, i] for i, name in enumerate(class_names)}
    })
    predictions_df.to_csv(
        os.path.join(save_dir, 'predictions.csv'),
        index=False
    )
    
    print(f"\nEvaluation complete! Results saved to {save_dir}")
    print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    
    return metrics


if __name__ == "__main__":
    # Test metrics functions
    print("Testing metrics functions...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_probs = np.random.rand(n_samples, n_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    
    class_names = ['Asthma', 'Bronchitis', 'COPD', 'Pneumonia', 'Healthy']
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs, class_names)
    
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Test plotting
    import os
    os.makedirs('test_results', exist_ok=True)
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path='test_results/test_cm.png'
    )
    
    plot_roc_curves(
        y_true,
        y_probs,
        class_names,
        save_path='test_results/test_roc.png'
    )
    
    plot_per_class_metrics(
        metrics,
        class_names,
        save_path='test_results/test_metrics.png'
    )
    
    save_metrics_report(
        metrics,
        class_names,
        'test_results/test_report.txt'
    )
    
    print("\nMetrics test complete! Check test_results/ directory.")
