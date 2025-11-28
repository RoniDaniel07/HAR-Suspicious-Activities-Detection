"""
Utility functions for HAR theft detection system
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple
import pandas as pd


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint with training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, filepath, optimizer=None, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Try to load state dict, with helpful error message
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"\nError loading checkpoint: {filepath}")
        print(f"Model architecture mismatch!")
        print(f"\nCheckpoint contains keys for a different model architecture.")
        print(f"Make sure you're using the same --model_name that was used during training.")
        print(f"\nOriginal error: {str(e)[:200]}...")
        raise
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def compute_metrics(y_true, y_pred, y_scores, class_names):
    """Compute comprehensive classification metrics"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics['per_class'] = report
    
    # ROC-AUC for binary or multi-class
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        metrics['roc_auc'] = float(auc(fpr, tpr))
        metrics['pr_auc'] = float(average_precision_score(y_true, y_scores[:, 1]))
    else:
        # Multi-class: compute one-vs-rest AUC
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        aucs = []
        for i in range(len(class_names)):
            if len(np.unique(y_true_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                aucs.append(auc(fpr, tpr))
        metrics['roc_auc_macro'] = float(np.mean(aucs)) if aucs else 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(y_true, y_scores, class_names, save_path):
    """Plot ROC curve (binary or multi-class)"""
    plt.figure(figsize=(10, 8))
    
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        for i in range(len(class_names)):
            if len(np.unique(y_true_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved: {save_path}")


def plot_training_curves(history, save_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # F1 Score
    if 'train_f1' in history:
        axes[1, 0].plot(history['train_f1'], label='Train')
        axes[1, 0].plot(history['val_f1'], label='Validation')
        axes[1, 0].set_title('F1 Score (Macro)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # Learning Rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")


def save_metrics_json(metrics, filepath):
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {filepath}")


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_class_weights(labels, num_classes):
    """Compute class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(num_classes)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.FloatTensor(weights)
