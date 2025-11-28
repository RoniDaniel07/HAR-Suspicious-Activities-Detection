"""
Training script for HAR theft detection models
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import VideoClipDataset
from src.models import build_model
from src.transforms import VideoTransform
from src.utils import (
    save_checkpoint, compute_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_training_curves, save_metrics_json, AverageMeter
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train HAR model for theft detection')
    
    # Data
    parser.add_argument('--train_csv', type=str, default='data/metadata_train.csv')
    parser.add_argument('--val_csv', type=str, default='data/metadata_val.csv')
    parser.add_argument('--clips_dir', type=str, default='data/clips')
    
    # Model
    parser.add_argument('--model_name', type=str, default='timesformer',
                        choices=['timesformer', 'videoswin', 'r3d', 'slowfast', 'simple3d'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/checkpoints')
    parser.add_argument('--log_dir', type=str, default='results/logs')
    parser.add_argument('--save_freq', type=int, default=5)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, args):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch in pbar:
        videos = batch['video'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if args.use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
        
        # Compute accuracy
        _, preds = outputs.max(1)
        acc = (preds == labels).float().mean()
        
        losses.update(loss.item(), videos.size(0))
        accs.update(acc.item(), videos.size(0))
        
        pbar.set_postfix({'loss': losses.avg, 'acc': accs.avg})
    
    return losses.avg, accs.avg


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, args):
    """Validate model"""
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    for batch in pbar:
        videos = batch['video'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Get predictions and scores
        scores = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        acc = (preds == labels).float().mean()
        
        losses.update(loss.item(), videos.size(0))
        accs.update(acc.item(), videos.size(0))
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())
        
        pbar.set_postfix({'loss': losses.avg, 'acc': accs.avg})
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    return losses.avg, accs.avg, all_preds, all_labels, all_scores


def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Training configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_transform = VideoTransform(mode='train')
    val_transform = VideoTransform(mode='val')
    
    train_dataset = VideoClipDataset(
        metadata_csv=args.train_csv,
        clips_dir=args.clips_dir,
        num_frames=args.num_frames,
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = VideoClipDataset(
        metadata_csv=args.val_csv,
        clips_dir=args.clips_dir,
        num_frames=args.num_frames,
        transform=val_transform,
        mode='val'
    )
    
    # Create dataloaders
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else args.num_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Build model
    print(f"\nBuilding model: {args.model_name}")
    num_classes = len(train_dataset.labels)
    model = build_model(
        args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        num_frames=args.num_frames,
        dropout=args.dropout
    )
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    if args.use_class_weights:
        class_weights = train_dataset.get_class_weights().to(args.device)
        print(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if args.use_amp and torch.cuda.is_available() else None
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    best_val_f1 = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, args.device, epoch, args
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_scores = validate(
            model, val_loader, criterion, args.device, epoch, args
        )
        
        # Compute detailed metrics
        metrics = compute_metrics(
            val_labels, val_preds, val_scores, train_dataset.labels
        )
        val_f1 = metrics['f1_macro']
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(args.output_dir, 'best_acc.pth')
            )
            print(f"  -> New best accuracy: {best_val_acc*100:.2f}%")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(args.output_dir, 'best_f1.pth')
            )
            print(f"  -> New best F1: {best_val_f1:.4f}")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs, metrics,
        os.path.join(args.output_dir, 'final.pth')
    )
    
    # Final evaluation
    print("\nFinal evaluation...")
    _, _, val_preds, val_labels, val_scores = validate(
        model, val_loader, criterion, args.device, args.epochs, args
    )
    
    final_metrics = compute_metrics(
        val_labels, val_preds, val_scores, train_dataset.labels
    )
    
    # Save metrics
    save_metrics_json(
        final_metrics,
        os.path.join('results/metrics', f'{args.model_name}_metrics.json')
    )
    
    # Plot results
    plot_confusion_matrix(
        val_labels, val_preds, train_dataset.labels,
        os.path.join('results/figures', f'{args.model_name}_confusion_matrix.png')
    )
    
    plot_roc_curve(
        val_labels, val_scores, train_dataset.labels,
        os.path.join('results/figures', f'{args.model_name}_roc_curve.png')
    )
    
    plot_training_curves(
        history,
        os.path.join('results/figures', f'{args.model_name}_training_curves.png')
    )
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"\nFinal Performance:")
    print(f"  Accuracy: {final_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score (Macro): {final_metrics['f1_macro']:.4f}")
    print(f"  Precision (Macro): {final_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {final_metrics['recall_macro']:.4f}")
    
    writer.close()


if __name__ == '__main__':
    main()
