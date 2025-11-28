"""
Extract clips from raw videos for training
"""
import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract clips from videos')
    
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with raw videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for clips')
    parser.add_argument('--clip_len', type=int, default=32, help='Frames per clip')
    parser.add_argument('--stride', type=int, default=16, help='Stride between clips')
    parser.add_argument('--fps', type=int, default=10, help='Target FPS')
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'mp4'], help='Output format')
    parser.add_argument('--min_clips', type=int, default=1, help='Minimum clips per video')
    
    return parser.parse_args()


def extract_frames_from_video(video_path, target_fps=10):
    """Extract frames from video at target FPS"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Cannot open {video_path}")
        return []
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, int(original_fps / target_fps))
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    return frames


def create_clips(frames, clip_length, stride):
    """Create overlapping clips from frames"""
    clips = []
    
    for start_idx in range(0, len(frames) - clip_length + 1, stride):
        clip = frames[start_idx:start_idx + clip_length]
        if len(clip) == clip_length:
            clips.append(np.array(clip))
    
    return clips


def save_clip(clip, output_path, format='npy'):
    """Save clip to disk"""
    if format == 'npy':
        np.save(output_path, clip)
    elif format == 'mp4':
        # Save as video
        height, width = clip.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))
        
        for frame in clip:
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()


def process_video(video_path, output_dir, args, video_id, label):
    """Process a single video and extract clips"""
    
    # Extract frames
    frames = extract_frames_from_video(video_path, target_fps=args.fps)
    
    if len(frames) < args.clip_len:
        print(f"Warning: Video {video_path} has only {len(frames)} frames, skipping")
        return []
    
    # Create clips
    clips = create_clips(frames, args.clip_len, args.stride)
    
    if len(clips) < args.min_clips:
        print(f"Warning: Video {video_path} produced only {len(clips)} clips, skipping")
        return []
    
    # Save clips and create metadata
    metadata = []
    
    for clip_idx, clip in enumerate(clips):
        # Create clip ID
        clip_id = f"{video_id}_clip_{clip_idx:04d}"
        
        if args.format == 'npy':
            clip_filename = f"{clip_id}.npy"
        else:
            clip_filename = f"{clip_id}.mp4"
        
        clip_path = os.path.join(output_dir, clip_filename)
        
        # Save clip
        save_clip(clip, clip_path, format=args.format)
        
        # Calculate timestamps
        start_frame = clip_idx * args.stride
        end_frame = start_frame + args.clip_len
        start_time = start_frame / args.fps
        end_time = end_frame / args.fps
        
        metadata.append({
            'clip_id': clip_filename,
            'video_id': video_id,
            'label': label,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time
        })
    
    return metadata


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all videos
    input_path = Path(args.input_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_path.rglob(f'*{ext}')))
    
    if len(video_files) == 0:
        print(f"No videos found in {args.input_dir}")
        return
    
    print(f"Found {len(video_files)} videos")
    
    # Process videos
    all_metadata = []
    
    for video_path in tqdm(video_files, desc='Processing videos'):
        # Infer label from directory structure
        # Assumes structure: input_dir/label/video.mp4
        relative_path = video_path.relative_to(input_path)
        
        if len(relative_path.parts) > 1:
            label = relative_path.parts[0]
        else:
            label = 'unknown'
        
        video_id = video_path.stem
        
        # Process video
        metadata = process_video(
            str(video_path),
            args.output_dir,
            args,
            video_id,
            label
        )
        
        all_metadata.extend(metadata)
    
    # Save metadata
    if len(all_metadata) > 0:
        df = pd.DataFrame(all_metadata)
        
        # Split into train/val
        train_df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=0.8, random_state=42)
        )
        val_df = df[~df.index.isin(train_df.index)]
        
        # Save CSVs
        output_path = Path(args.output_dir).parent
        train_csv = output_path / 'metadata_train.csv'
        val_csv = output_path / 'metadata_val.csv'
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        
        print(f"\nExtraction completed!")
        print(f"Total clips: {len(df)}")
        print(f"Train clips: {len(train_df)} -> {train_csv}")
        print(f"Val clips: {len(val_df)} -> {val_csv}")
        print(f"\nClass distribution:")
        print(df['label'].value_counts())
    else:
        print("No clips extracted!")


if __name__ == '__main__':
    main()
