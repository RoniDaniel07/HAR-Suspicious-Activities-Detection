"""
Dataset classes for HAR theft detection
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional
from src.transforms import VideoTransform, load_video_frames


class VideoClipDataset(Dataset):
    """Dataset for video clips with labels"""
    
    def __init__(
        self,
        metadata_csv: str,
        clips_dir: str,
        num_frames: int = 32,
        transform: Optional[VideoTransform] = None,
        mode: str = 'train'
    ):
        """
        Args:
            metadata_csv: path to CSV with columns [clip_id, video_id, label, ...]
            clips_dir: directory containing video clips
            num_frames: number of frames per clip
            transform: VideoTransform instance
            mode: 'train', 'val', or 'test'
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.clips_dir = clips_dir
        self.num_frames = num_frames
        self.transform = transform or VideoTransform(mode=mode)
        self.mode = mode
        
        # Get unique labels and create mapping
        self.labels = sorted(self.metadata['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Loaded {len(self.metadata)} clips from {metadata_csv}")
        print(f"Classes: {self.labels}")
        print(f"Class distribution:")
        print(self.metadata['label'].value_counts())
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load video clip
        clip_path = os.path.join(self.clips_dir, row['clip_id'])
        
        # Handle different file formats
        if not os.path.exists(clip_path):
            # Try with extensions
            for ext in ['.mp4', '.avi', '.npy']:
                test_path = clip_path + ext
                if os.path.exists(test_path):
                    clip_path = test_path
                    break
        
        # Load frames
        if clip_path.endswith('.npy'):
            # Pre-extracted frames as numpy array
            frames = np.load(clip_path)
        else:
            # Load from video file
            frames = load_video_frames(clip_path, num_frames=self.num_frames)
        
        # Ensure we have exactly num_frames
        if len(frames) < self.num_frames:
            # Pad with last frame if needed
            while len(frames) < self.num_frames:
                frames = np.concatenate([frames, frames[-1:]], axis=0)
        elif len(frames) > self.num_frames:
            # Trim to exact size
            frames = frames[:self.num_frames]
        
        # Apply transforms
        video_tensor = self.transform(frames)
        
        # Final safety check - ensure correct size
        if video_tensor.shape[0] != self.num_frames:
            # If transform changed size, resample to correct size
            indices = np.linspace(0, video_tensor.shape[0]-1, self.num_frames, dtype=int)
            video_tensor = video_tensor[indices]
        
        # Get label
        label = self.label_to_idx[row['label']]
        
        return {
            'video': video_tensor,
            'label': label,
            'clip_id': row['clip_id']
        }
    
    def get_class_weights(self):
        """Compute class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        labels = self.metadata['label'].map(self.label_to_idx).values
        weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(self.labels)),
            y=labels
        )
        return torch.FloatTensor(weights)


class VideoDataset(Dataset):
    """Dataset for full videos (for inference)"""
    
    def __init__(
        self,
        video_paths: List[str],
        clip_length: int = 32,
        stride: int = 16,
        fps: int = 10,
        transform: Optional[VideoTransform] = None
    ):
        """
        Args:
            video_paths: list of video file paths
            clip_length: frames per clip
            stride: stride between clips
            fps: target fps
            transform: VideoTransform instance
        """
        self.video_paths = video_paths
        self.clip_length = clip_length
        self.stride = stride
        self.fps = fps
        self.transform = transform or VideoTransform(mode='test')
        
        # Pre-compute clip information
        self.clips_info = []
        for video_path in video_paths:
            clips = self._get_video_clips(video_path)
            self.clips_info.extend(clips)
    
    def _get_video_clips(self, video_path):
        """Get all clips from a video"""
        from src.transforms import extract_clips_from_video
        
        clips = extract_clips_from_video(
            video_path,
            clip_length=self.clip_length,
            stride=self.stride,
            fps=self.fps
        )
        
        return [
            {
                'video_path': video_path,
                'clip_idx': i,
                'frames': clip
            }
            for i, clip in enumerate(clips)
        ]
    
    def __len__(self):
        return len(self.clips_info)
    
    def __getitem__(self, idx):
        clip_info = self.clips_info[idx]
        frames = clip_info['frames']
        
        # Apply transform
        video_tensor = self.transform(frames)
        
        return {
            'video': video_tensor,
            'video_path': clip_info['video_path'],
            'clip_idx': clip_info['clip_idx']
        }


def create_dummy_dataset(output_dir='data', num_videos_per_class=5, num_frames=32, classes=None):
    """
    Create a small dummy dataset for quick testing
    
    Args:
        output_dir: where to save the dataset
        num_videos_per_class: number of videos per class
        num_frames: frames per video
        classes: list of class names (default: ['normal', 'suspicious', 'theft'])
    """
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    clips_dir = os.path.join(output_dir, 'clips')
    os.makedirs(clips_dir, exist_ok=True)
    
    # Default to 3 classes, or use extended 6 classes
    if classes is None:
        classes = ['normal', 'suspicious', 'theft']
    
    metadata = []
    
    print("Creating dummy dataset...")
    
    for class_name in classes:
        for i in range(num_videos_per_class):
            clip_id = f"{class_name}_{i:03d}.npy"
            clip_path = os.path.join(clips_dir, clip_id)
            
            # Generate random frames (simulating video)
            # Different patterns for different classes
            if class_name == 'normal':
                # Slow moving patterns
                frames = np.random.randint(100, 150, (num_frames, 224, 224, 3), dtype=np.uint8)
            elif class_name == 'suspicious':
                # Medium intensity with some variation
                frames = np.random.randint(80, 180, (num_frames, 224, 224, 3), dtype=np.uint8)
            elif class_name == 'theft':
                # High variation, darker
                frames = np.random.randint(50, 200, (num_frames, 224, 224, 3), dtype=np.uint8)
            elif class_name == 'fighting':
                # High motion, rapid changes
                frames = np.random.randint(60, 220, (num_frames, 224, 224, 3), dtype=np.uint8)
            elif class_name == 'weapon_gun':
                # Darker, focused patterns
                frames = np.random.randint(40, 160, (num_frames, 224, 224, 3), dtype=np.uint8)
            elif class_name == 'weapon_other':
                # Similar to weapon_gun
                frames = np.random.randint(45, 165, (num_frames, 224, 224, 3), dtype=np.uint8)
            else:
                # Default pattern
                frames = np.random.randint(50, 200, (num_frames, 224, 224, 3), dtype=np.uint8)
            
            # Save as numpy array
            np.save(clip_path, frames)
            
            metadata.append({
                'clip_id': clip_id,
                'video_id': f"{class_name}_video_{i}",
                'label': class_name,
                'start_frame': 0,
                'end_frame': num_frames,
                'start_time': 0.0,
                'end_time': num_frames / 10.0
            })
    
    # Create train/val split
    df = pd.DataFrame(metadata)
    
    # 80-20 split
    train_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(frac=0.8, random_state=42)
    )
    val_df = df[~df.index.isin(train_df.index)]
    
    # Save metadata
    train_csv = os.path.join(output_dir, 'metadata_train.csv')
    val_csv = os.path.join(output_dir, 'metadata_val.csv')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"Created dummy dataset:")
    print(f"  Train: {len(train_df)} clips -> {train_csv}")
    print(f"  Val: {len(val_df)} clips -> {val_csv}")
    print(f"  Clips saved in: {clips_dir}")
    
    return train_csv, val_csv


if __name__ == '__main__':
    # Test dataset creation
    train_csv, val_csv = create_dummy_dataset()
    
    # Test dataset loading
    from torch.utils.data import DataLoader
    
    dataset = VideoClipDataset(
        metadata_csv=train_csv,
        clips_dir='data/clips',
        num_frames=32,
        mode='train'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Classes: {dataset.labels}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample video shape: {sample['video'].shape}")
    print(f"Sample label: {sample['label']}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch video shape: {batch['video'].shape}")
    print(f"Batch labels: {batch['label']}")
