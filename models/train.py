"""
Training script for RespiScope-AI
Handles complete training pipeline with validation and checkpointing
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config, save_config
from utils.metrics import calculate_metrics, plot_confusion_matrix
from crnn_classifier import create_model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=15, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config, model, train_loader, val_loader, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        if config.training.loss_fn == 'focal_loss':
            self.criterion = FocalLoss(
                alpha=config.training.focal_alpha,
                gamma=config.training.focal_gamma
            )
        else:
            # Use class weights if specified
            if config.training.use_class_weights:
                class_weights = self._compute_class_weights()
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        if config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.num_epochs
            )
        elif config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif config.training.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.training.scheduler_patience,
                factor=0.5
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='min'
        )
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.training.use_amp else None
        
        # TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.log_dir, config.experiment_name)
        )
        
        # Tracking
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _compute_class_weights(self):
        """Compute class weights for imbalanced datasets"""
        # This should be computed from your dataset
        # For now, return uniform weights
        return torch.ones(self.config.model.num_classes).to(self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1} [Train]')
        
        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(audio)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(audio)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch + 1} [Val]')
            
            for audio, labels in pbar:
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(audio)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            self.config.model.class_names
        )
        
        return epoch_loss, metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{self.epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model: {self.config.model.model_type}")
        print(f"Epochs: {self.config.training.num_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/val', val_metrics['weighted_f1'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Val F1: {val_metrics['weighted_f1']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_rate', current_lr, epoch)
                print(f"Learning rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Close writer
        self.writer.close()
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        history_path = os.path.join(
            self.config.checkpoint_dir,
            'training_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Train RespiScope-AI model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset splits')
    parser.add_argument('--model_type', type=str, default='crnn', choices=['crnn', 'transformer'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='models/checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        from utils.config import load_config
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Update config with command line arguments
    config.data.splits_dir = args.data_path
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.checkpoint_dir = args.save_dir
    config.augmentation.enabled = args.augment
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Save config
    save_config(config, os.path.join(config.checkpoint_dir, 'config.yaml'))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = str(device)
    
    # TODO: Load datasets (implement dataset class)
    # For now, using dummy data
    print("Loading datasets...")
    # train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=args.model_type,
        num_classes=config.model.num_classes,
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # TODO: Create trainer and train
    # trainer = Trainer(config, model, train_loader, val_loader, device)
    # trainer.train()
    
    print("\nTraining script ready! Please implement dataset loading.")


if __name__ == '__main__':
    main()
