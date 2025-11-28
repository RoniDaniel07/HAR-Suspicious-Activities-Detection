"""
HAR Theft Detection System
High-accuracy video activity recognition for surveillance
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.models import build_model
from src.datasets import VideoClipDataset, create_dummy_dataset
from src.transforms import VideoTransform
from src.utils import compute_metrics, save_checkpoint, load_checkpoint

__all__ = [
    'build_model',
    'VideoClipDataset',
    'create_dummy_dataset',
    'VideoTransform',
    'compute_metrics',
    'save_checkpoint',
    'load_checkpoint'
]
