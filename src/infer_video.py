"""
Inference on video files - detect theft/suspicious activities
"""
import os
import sys
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.transforms import VideoTransform, extract_clips_from_video
from src.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on video')
    
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--model_name', type=str, default='timesformer')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--clip_length', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--class_names', type=str, nargs='+', default=['normal', 'suspicious', 'theft'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_video', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    
    return parser.parse_args()


@torch.no_grad()
def predict_video(video_path, model, transform, args):
    """Run inference on video and return predictions per clip"""
    
    print(f"Extracting clips from video: {video_path}")
    clips = extract_clips_from_video(
        video_path,
        clip_length=args.clip_length,
        stride=args.stride,
        fps=args.fps
    )
    
    if len(clips) == 0:
        print("Warning: No clips extracted from video")
        return []
    
    print(f"Extracted {len(clips)} clips")
    
    model.eval()
    predictions = []
    
    for i, clip_frames in enumerate(tqdm(clips, desc='Processing clips')):
        # Transform clip
        video_tensor = transform(clip_frames)
        video_tensor = video_tensor.unsqueeze(0).to(args.device)  # Add batch dimension
        
        # Predict
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        
        # Calculate timestamps
        start_frame = i * args.stride
        end_frame = start_frame + args.clip_length
        start_time = start_frame / args.fps
        end_time = end_frame / args.fps
        
        predictions.append({
            'clip_idx': i,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'predicted_class': pred_class,
            'predicted_label': args.class_names[pred_class],
            'confidence': confidence,
            'probs': probs.cpu().numpy()
        })
    
    return predictions


def create_output_video(input_video, predictions, output_path, args):
    """Create output video with predictions overlaid"""
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create frame-to-prediction mapping
    frame_preds = {}
    for pred in predictions:
        for frame_idx in range(pred['start_frame'], pred['end_frame']):
            if frame_idx not in frame_preds:
                frame_preds[frame_idx] = []
            frame_preds[frame_idx].append(pred)
    
    print(f"Creating output video: {output_path}")
    frame_idx = 0
    
    with tqdm(total=total_frames, desc='Writing video') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get predictions for this frame
            if frame_idx in frame_preds:
                preds = frame_preds[frame_idx]
                
                # Use the prediction with highest confidence
                best_pred = max(preds, key=lambda x: x['confidence'])
                
                label = best_pred['predicted_label']
                conf = best_pred['confidence']
                
                # Choose color based on label
                if label == 'theft':
                    color = (0, 0, 255)  # Red
                elif label == 'suspicious':
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green
                
                # Draw rectangle and text
                cv2.rectangle(frame, (10, 10), (width - 10, 80), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (width - 10, 80), color, 3)
                
                text = f"{label.upper()}: {conf*100:.1f}%"
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1.2, color, 3)
                
                # Add timestamp
                time_text = f"Time: {frame_idx/fps:.1f}s"
                cv2.putText(frame, time_text, (20, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"Output video saved: {output_path}")


def main():
    args = parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Load model
    print(f"Loading model: {args.model_name}")
    num_classes = len(args.class_names)
    model = build_model(args.model_name, num_classes=num_classes, pretrained=False)
    load_checkpoint(model, args.checkpoint, device=args.device)
    model = model.to(args.device)
    model.eval()
    
    # Create transform
    transform = VideoTransform(mode='test')
    
    # Run inference
    predictions = predict_video(args.video, model, transform, args)
    
    if len(predictions) == 0:
        print("No predictions generated")
        return
    
    # Filter detections by threshold (for suspicious/theft)
    detections = []
    for pred in predictions:
        if pred['predicted_label'] != 'normal' and pred['confidence'] >= args.threshold:
            detections.append(pred)
    
    print(f"\n{'='*60}")
    print(f"DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Total clips analyzed: {len(predictions)}")
    print(f"Suspicious/Theft detections: {len(detections)}")
    
    if len(detections) > 0:
        print(f"\nDetected activities:")
        for det in detections:
            print(f"  [{det['start_time']:.1f}s - {det['end_time']:.1f}s] "
                  f"{det['predicted_label'].upper()} (confidence: {det['confidence']*100:.1f}%)")
    
    # Save predictions to CSV
    if args.output_csv:
        df = pd.DataFrame(predictions)
        df.to_csv(args.output_csv, index=False)
        print(f"\nPredictions saved to: {args.output_csv}")
    
    # Create output video
    if args.output_video:
        create_output_video(args.video, predictions, args.output_video, args)
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()
