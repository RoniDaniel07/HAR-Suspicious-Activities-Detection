"""
Video transforms and augmentations for HAR
"""
import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Dict
import cv2


class VideoTransform:
    """Video-specific transforms"""
    
    def __init__(self, mode='train', img_size=224, mean=None, std=None):
        self.mode = mode
        self.img_size = img_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        if mode == 'train':
            self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
    
    def __call__(self, video_frames):
        """
        Args:
            video_frames: numpy array of shape (T, H, W, C) or list of frames
        Returns:
            tensor of shape (T, C, H, W)
        """
        if isinstance(video_frames, list):
            video_frames = np.stack(video_frames)
        
        # Apply spatial transform to each frame
        transformed_frames = []
        for frame in video_frames:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            transformed = self.spatial_transform(frame)
            transformed_frames.append(transformed)
        
        # Stack to (T, C, H, W)
        video_tensor = torch.stack(transformed_frames)
        
        # Optional temporal augmentation for training
        if self.mode == 'train':
            video_tensor = self.temporal_augment(video_tensor)
        
        return video_tensor
    
    def temporal_augment(self, video_tensor):
        """Apply temporal augmentations"""
        # Note: Temporal augmentation disabled to maintain consistent tensor sizes
        # Random temporal sampling can cause size mismatches in batches
        # If needed, implement with padding/interpolation to maintain size
        return video_tensor


def load_video_frames(video_path, num_frames=32, fps=None):
    """
    Load video and extract frames
    Args:
        video_path: path to video file
        num_frames: number of frames to extract
        fps: target fps (if None, use original fps)
    Returns:
        numpy array of shape (num_frames, H, W, C)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract
    if fps is not None:
        frame_step = max(1, int(original_fps / fps))
    else:
        frame_step = max(1, total_frames // num_frames)
    
    frames = []
    frame_idx = 0
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    # Handle cases where we got fewer frames
    if len(frames) < num_frames:
        # Repeat last frame
        while len(frames) < num_frames:
            frames.append(frames[-1])
    elif len(frames) > num_frames:
        # Uniformly sample
        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    return np.array(frames)


def extract_clips_from_video(video_path, clip_length=32, stride=16, fps=10):
    """
    Extract overlapping clips from a video
    Args:
        video_path: path to video
        clip_length: number of frames per clip
        stride: stride between clips
        fps: target fps
    Returns:
        list of clips, each clip is numpy array (clip_length, H, W, C)
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, int(original_fps / fps))
    
    all_frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    # Create clips with sliding window
    clips = []
    for start_idx in range(0, len(all_frames) - clip_length + 1, stride):
        clip = all_frames[start_idx:start_idx + clip_length]
        if len(clip) == clip_length:
            clips.append(np.array(clip))
    
    return clips
