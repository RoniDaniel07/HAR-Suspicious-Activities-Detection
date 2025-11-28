"""
Model architectures for HAR theft detection
Supports: TimeSformer, Video Swin, SlowFast, R3D, X3D, and Ensemble
"""
import torch
import torch.nn as nn
import timm
from typing import Optional


class TimeSformerModel(nn.Module):
    """TimeSformer video transformer model"""
    
    def __init__(self, num_classes=3, pretrained=True, img_size=224, num_frames=32, dropout=0.3):
        super().__init__()
        self.num_frames = num_frames
        
        # Use ViT as base and adapt for video
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            img_size=img_size
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Temporal embedding for frames
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, self.feature_dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames
        x = x.view(B * T, C, H, W)
        
        # Extract spatial features
        features = self.backbone.forward_features(x)  # (B*T, num_patches+1, feature_dim)
        
        # Take CLS token
        cls_tokens = features[:, 0]  # (B*T, feature_dim)
        
        # Reshape back to (B, T, feature_dim)
        cls_tokens = cls_tokens.view(B, T, self.feature_dim)
        
        # Add temporal embeddings
        cls_tokens = cls_tokens + self.temporal_embed[:, :T, :]
        
        # Temporal pooling (mean over time)
        video_features = cls_tokens.mean(dim=1)  # (B, feature_dim)
        
        # Classification
        logits = self.classifier(video_features)
        
        return logits


class VideoSwinModel(nn.Module):
    """Video Swin Transformer (simplified version using 2D Swin + temporal pooling)"""
    
    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        super().__init__()
        
        # Use Swin Transformer as backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.feature_dim = self.backbone.num_features
        
        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Process frames
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, feature_dim)
        
        # Reshape to (B, T, feature_dim)
        features = features.view(B, T, self.feature_dim)
        
        # Temporal attention
        attn_out, _ = self.temporal_attn(features, features, features)
        
        # Temporal pooling
        video_features = attn_out.mean(dim=1)  # (B, feature_dim)
        
        # Classification
        logits = self.classifier(video_features)
        
        return logits


class R3DModel(nn.Module):
    """3D ResNet model"""
    
    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        super().__init__()
        
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            
            if pretrained:
                self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            else:
                self.backbone = r3d_18(weights=None)
            
            # Replace final FC layer
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        except ImportError:
            # Fallback: simple 3D CNN
            self.backbone = Simple3DCNN(num_classes, dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        # Rearrange to (B, C, T, H, W) for 3D conv
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)


class Simple3DCNN(nn.Module):
    """Simple 3D CNN fallback"""
    
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Input: (B, T, C, H, W) -> need (B, C, T, H, W) for Conv3d
        if x.dim() == 5 and x.shape[2] == 3:  # Check if in (B, T, C, H, W) format
            x = x.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SlowFastModel(nn.Module):
    """SlowFast model (simplified dual-pathway)"""
    
    def __init__(self, num_classes=3, pretrained=False, dropout=0.3):
        super().__init__()
        
        # Slow pathway (low frame rate, high spatial resolution)
        self.slow_pathway = Simple3DCNN(num_classes=512, dropout=0.1)
        
        # Fast pathway (high frame rate, low spatial resolution)
        self.fast_pathway = Simple3DCNN(num_classes=128, dropout=0.1)
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Slow pathway: sample every 4th frame
        slow_indices = torch.arange(0, T, 4, device=x.device)
        x_slow = x[:, slow_indices]
        x_slow = x_slow.permute(0, 2, 1, 3, 4)
        
        # Fast pathway: all frames, downsampled spatially
        x_fast = nn.functional.interpolate(
            x.view(B * T, C, H, W),
            size=(H // 2, W // 2),
            mode='bilinear',
            align_corners=False
        )
        x_fast = x_fast.view(B, T, C, H // 2, W // 2)
        x_fast = x_fast.permute(0, 2, 1, 3, 4)
        
        # Extract features
        slow_feat = self.slow_pathway(x_slow)
        fast_feat = self.fast_pathway(x_fast)
        
        # Concatenate and classify
        combined = torch.cat([slow_feat, fast_feat], dim=1)
        logits = self.fusion(combined)
        
        return logits


class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(self, models, num_classes=3, fusion='average'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.fusion = fusion
        self.num_classes = num_classes
        
        if fusion == 'learned':
            # Learnable fusion weights
            self.fusion_weights = nn.Parameter(torch.ones(len(models)) / len(models))
            self.fusion_mlp = nn.Sequential(
                nn.Linear(num_classes * len(models), 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        outputs = []
        
        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                out = model(x)
                outputs.append(out)
        
        if self.fusion == 'average':
            # Simple averaging
            logits = torch.stack(outputs).mean(dim=0)
        elif self.fusion == 'learned':
            # Learned fusion
            stacked = torch.cat(outputs, dim=1)
            logits = self.fusion_mlp(stacked)
        else:
            # Weighted average
            weights = torch.softmax(self.fusion_weights, dim=0)
            logits = sum(w * out for w, out in zip(weights, outputs))
        
        return logits


def build_model(model_name: str, num_classes: int = 3, pretrained: bool = True, **kwargs):
    """
    Build model by name
    
    Args:
        model_name: one of ['timesformer', 'videoswin', 'r3d', 'slowfast', 'simple3d']
        num_classes: number of output classes
        pretrained: use pretrained weights
        **kwargs: additional model-specific arguments
    
    Returns:
        nn.Module
    """
    model_name = model_name.lower()
    
    if model_name == 'timesformer':
        model = TimeSformerModel(
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=kwargs.get('num_frames', 32),
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_name == 'videoswin':
        model = VideoSwinModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_name == 'r3d' or model_name == 'r3d_18':
        model = R3DModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_name == 'slowfast':
        model = SlowFastModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_name == 'simple3d':
        model = Simple3DCNN(
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


if __name__ == '__main__':
    # Test models
    batch_size = 2
    num_frames = 32
    x = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    print("Testing TimeSformer...")
    model = build_model('timesformer', num_classes=3, num_frames=num_frames)
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    print("\nTesting SlowFast...")
    model = build_model('slowfast', num_classes=3)
    out = model(x)
    print(f"Output shape: {out.shape}")
