"""
Evaluation script for HAR models
"""
import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import VideoClipDataset
from src.models import build_model, EnsembleModel
from src.transforms import VideoTransform
from src.utils import (
    load_checkpoint, compute_metrics, plot_confusion_matrix,
    plot_roc_curve, save_metrics_json
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HAR model')
    
    parser.add_argument('--test_csv', type=str, default='data/metadata_val.csv')
    parser.add_argument('--clips_dir', type=str, default='data/clips')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='timesformer')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results/eval')
    
    # Ensemble options
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--checkpoint2', type=str, default=None)
    parser.add_argument('--model_name2', type=str, default=None)
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    all_clip_ids = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        videos = batch['video'].to(device)
        labels = batch['label']
        
        outputs = model(videos)
        scores = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_scores.extend(scores.cpu().numpy())
        all_clip_ids.extend(batch['clip_id'])
    
    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_scores),
        all_clip_ids
    )


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading dataset...")
    transform = VideoTransform(mode='test')
    
    dataset = VideoClipDataset(
        metadata_csv=args.test_csv,
        clips_dir=args.clips_dir,
        num_frames=args.num_frames,
        transform=transform,
        mode='test'
    )
    
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else args.num_workers
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    num_classes = len(dataset.labels)
    class_names = dataset.labels
    
    # Load model(s)
    if args.ensemble and args.checkpoint2:
        print("Loading ensemble models...")
        
        # Model 1
        model1 = build_model(args.model_name, num_classes=num_classes, pretrained=False)
        load_checkpoint(model1, args.checkpoint, device=args.device)
        model1 = model1.to(args.device)
        
        # Model 2
        model2 = build_model(args.model_name2, num_classes=num_classes, pretrained=False)
        load_checkpoint(model2, args.checkpoint2, device=args.device)
        model2 = model2.to(args.device)
        
        # Create ensemble
        model = EnsembleModel([model1, model2], num_classes=num_classes, fusion='average')
        model = model.to(args.device)
        
        model_name = f"ensemble_{args.model_name}_{args.model_name2}"
    else:
        print(f"Loading model: {args.model_name}")
        model = build_model(args.model_name, num_classes=num_classes, pretrained=False)
        load_checkpoint(model, args.checkpoint, device=args.device)
        model = model.to(args.device)
        model_name = args.model_name
    
    # Evaluate
    print("Evaluating...")
    preds, labels, scores, clip_ids = evaluate_model(model, dataloader, args.device)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(labels, preds, scores, class_names)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1 Score (Macro): {metrics['f1_macro']*100:.2f}%")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']*100:.2f}%")
    print(f"Precision (Macro): {metrics['precision_macro']*100:.2f}%")
    print(f"Recall (Macro): {metrics['recall_macro']*100:.2f}%")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']*100:.2f}%")
    if 'roc_auc_macro' in metrics:
        print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']*100:.2f}%")
    
    print("\nPer-class metrics:")
    for class_name in class_names:
        if class_name in metrics['per_class']:
            class_metrics = metrics['per_class'][class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']*100:.2f}%")
            print(f"    Recall: {class_metrics['recall']*100:.2f}%")
            print(f"    F1-Score: {class_metrics['f1-score']*100:.2f}%")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'{model_name}_metrics.json')
    save_metrics_json(metrics, metrics_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(labels, preds, class_names, cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(args.output_dir, f'{model_name}_roc_curve.png')
    plot_roc_curve(labels, scores, class_names, roc_path)
    
    # Save predictions
    import pandas as pd
    results_df = pd.DataFrame({
        'clip_id': clip_ids,
        'true_label': [class_names[l] for l in labels],
        'pred_label': [class_names[p] for p in preds],
        'correct': labels == preds
    })
    
    # Add probability scores
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = scores[:, i]
    
    results_path = os.path.join(args.output_dir, f'{model_name}_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nPredictions saved to: {results_path}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
