# Methodology: HAR System for Theft Detection

## 1. Problem Statement

### 1.1 Objective
Develop a high-accuracy Human Activity Recognition (HAR) system capable of detecting theft and suspicious activities in public places from CCTV surveillance footage in real-time.

### 1.2 Challenges
- **Temporal complexity**: Activities unfold over time, requiring temporal modeling
- **Class imbalance**: Normal activities are far more common than theft
- **Subtle differences**: Suspicious behavior may look similar to normal behavior
- **Real-time requirements**: System must process video streams efficiently
- **Varying conditions**: Different lighting, camera angles, and environments

### 1.3 Success Criteria
- Accuracy > 85% on test set
- F1-score > 0.80 for theft/suspicious classes
- Real-time inference capability (>10 FPS)
- Low false positive rate to avoid alert fatigue

## 2. Dataset

### 2.1 Data Sources

#### Primary Dataset: UCF-Crime
- Large-scale anomaly detection dataset
- 1,900 long untrimmed surveillance videos
- 13 anomaly categories including robbery, stealing, burglary
- Real-world CCTV footage

#### Secondary Datasets (Optional)
- Custom surveillance footage
- CamNuvem robbery dataset
- CCTV-Fights dataset

### 2.2 Data Preprocessing

#### Video to Clips Conversion
```
Input: Long surveillance videos (minutes to hours)
Output: Fixed-length clips (32 frames each)

Process:
1. Extract frames at target FPS (10 fps)
2. Create overlapping clips with sliding window
   - Clip length: 32 frames (~3.2 seconds)
   - Stride: 16 frames (50% overlap)
3. Resize frames to 224×224 pixels
4. Save clips with metadata (timestamps, labels)
```

#### Frame Preprocessing
- **Spatial**: Resize to 224×224, center crop
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Color space**: RGB

#### Data Augmentation (Training Only)
- Random horizontal flip (p=0.5)
- Random resized crop (scale=0.8-1.0)
- Color jitter (brightness, contrast, saturation, hue)
- Random temporal sampling (drop frames)

### 2.3 Class Definition

| Class | Description | Examples |
|-------|-------------|----------|
| Normal | Regular activities | Walking, standing, talking |
| Suspicious | Potentially concerning behavior | Loitering, looking around nervously, following someone |
| Theft | Criminal activities | Stealing, robbery, burglary, pickpocketing |

### 2.4 Data Split
- Training: 70%
- Validation: 15%
- Test: 15%

Stratified split to maintain class distribution.

## 3. Model Architectures

### 3.1 TimeSformer (Primary Model)

#### Architecture
```
Input: (B, T, C, H, W) = (batch, 32, 3, 224, 224)

1. Spatial Feature Extraction:
   - Vision Transformer (ViT-Base)
   - Patch size: 16×16
   - Embedding dim: 768
   - Process each frame independently
   
2. Temporal Modeling:
   - Temporal position embeddings
   - Temporal attention across frames
   - Aggregate frame features
   
3. Classification Head:
   - LayerNorm
   - Dropout (0.3)
   - Linear(768 → num_classes)

Output: (B, 3) class logits
```

#### Key Features
- Pretrained on Kinetics-400 (400 action classes)
- Divided space-time attention
- Efficient temporal modeling
- State-of-the-art accuracy

#### Parameters
- Total: ~87M parameters
- Trainable: ~87M (full fine-tuning)

### 3.2 Video Swin Transformer

#### Architecture
```
Input: (B, T, C, H, W)

1. Spatial Feature Extraction:
   - Swin Transformer backbone
   - Hierarchical attention
   - Shifted windows
   
2. Temporal Attention:
   - Multi-head attention across time
   - 8 attention heads
   
3. Classification Head:
   - LayerNorm + Dropout + Linear

Output: (B, 3) class logits
```

#### Key Features
- Hierarchical vision transformer
- Efficient attention mechanism
- Good accuracy-speed tradeoff

### 3.3 SlowFast (3D CNN)

#### Architecture
```
Input: (B, C, T, H, W)

Slow Pathway:
- Low temporal resolution (T/4 frames)
- High spatial resolution (224×224)
- 3D convolutions
- Captures spatial semantics

Fast Pathway:
- High temporal resolution (T frames)
- Low spatial resolution (112×112)
- 3D convolutions
- Captures motion

Fusion:
- Concatenate features
- MLP classifier

Output: (B, 3) class logits
```

#### Key Features
- Dual-pathway design
- Efficient motion modeling
- Fast inference

### 3.4 Simple 3D CNN (Baseline)

#### Architecture
```
Input: (B, C, T, H, W)

Conv3D Blocks:
1. Conv3D(3→64) + BN + ReLU + MaxPool
2. Conv3D(64→128) + BN + ReLU + MaxPool
3. Conv3D(128→256) + BN + ReLU + MaxPool
4. Conv3D(256→512) + BN + ReLU + AdaptiveAvgPool

Classifier:
- Dropout(0.3)
- Linear(512 → 3)

Output: (B, 3) class logits
```

#### Key Features
- Simple and fast
- Good baseline
- Easy to train

### 3.5 Ensemble Model

#### Strategy
```
Models: [TimeSformer, SlowFast]

Fusion Methods:
1. Average: mean(logits)
2. Weighted: Σ(w_i * logits_i)
3. Learned: MLP(concat(logits))

Output: (B, 3) class logits
```

#### Benefits
- Higher accuracy
- More robust predictions
- Combines complementary strengths

## 4. Training Strategy

### 4.1 Loss Function

#### Cross-Entropy with Enhancements
```python
loss = CrossEntropyLoss(
    weight=class_weights,      # Handle imbalance
    label_smoothing=0.1        # Regularization
)
```

#### Class Weights
Computed using sklearn's `compute_class_weight`:
```
weight_i = n_samples / (n_classes * n_samples_i)
```

### 4.2 Optimization

#### Optimizer: AdamW
```python
optimizer = AdamW(
    params=model.parameters(),
    lr=1e-4,              # TimeSformer
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)
```

#### Learning Rate Schedule
```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)
```

#### Gradient Clipping
```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4.3 Training Procedure

```
For each epoch:
    For each batch:
        1. Load video clips and labels
        2. Forward pass (with mixed precision)
        3. Compute loss
        4. Backward pass
        5. Clip gradients
        6. Optimizer step
        7. Update metrics
    
    Validation:
        1. Evaluate on validation set
        2. Compute metrics (accuracy, F1, AUC)
        3. Save best model
        4. Update learning rate
```

### 4.4 Hyperparameters

| Parameter | TimeSformer | SlowFast | Simple3D |
|-----------|-------------|----------|----------|
| Batch Size | 4 | 8 | 16 |
| Learning Rate | 1e-4 | 3e-4 | 3e-4 |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| Epochs | 25 | 25 | 15 |
| Warmup Epochs | 2 | 2 | 1 |
| Dropout | 0.3 | 0.3 | 0.3 |
| Label Smoothing | 0.1 | 0.1 | 0.1 |

### 4.5 Regularization Techniques

1. **Dropout**: 0.3 in classification head
2. **Label Smoothing**: 0.1
3. **Weight Decay**: 1e-4
4. **Data Augmentation**: Spatial and temporal
5. **Early Stopping**: Based on validation F1

### 4.6 Mixed Precision Training

```python
scaler = GradScaler()

with autocast():
    outputs = model(videos)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Benefits:
- 2x faster training
- 50% less GPU memory
- Minimal accuracy loss

## 5. Evaluation Metrics

### 5.1 Classification Metrics

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### Precision, Recall, F1-Score
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Computed per-class and averaged (macro, weighted).

#### Confusion Matrix
Visualizes classification performance across all classes.

### 5.2 Anomaly Detection Metrics

#### ROC-AUC
Area under ROC curve (normal vs. anomaly).

#### PR-AUC
Area under precision-recall curve.

### 5.3 Video-Level Metrics

For full video evaluation:
```
Video Score = max(clip_scores)  or  mean(top_k_scores)

If Video Score > threshold:
    Predict: Anomaly
Else:
    Predict: Normal
```

## 6. Inference Pipeline

### 6.1 Video Inference

```
Input: Video file

1. Extract clips:
   - Sliding window (stride=16)
   - 32 frames per clip
   
2. For each clip:
   - Preprocess frames
   - Run model inference
   - Get class probabilities
   
3. Post-processing:
   - Temporal smoothing (moving average)
   - Threshold detection
   - Generate timestamps
   
4. Output:
   - Detection CSV (timestamps, labels, scores)
   - Annotated video

Output: Detections with timestamps
```

### 6.2 Real-time Webcam

```
Initialize:
- Frame buffer (32 frames)
- Prediction history (5 predictions)

Loop:
    1. Capture frame
    2. Add to buffer
    3. If buffer full and time to infer:
        - Run inference
        - Add prediction to history
        - Smooth predictions (majority vote)
    4. Display current prediction
    5. Alert if suspicious/theft detected

Controls: 'q' to quit, 's' to screenshot
```

### 6.3 Temporal Smoothing

```python
# Moving average over last N predictions
smoothed_score = mean(scores[-N:])

# Majority voting
smoothed_class = mode(predictions[-N:])
```

Benefits:
- Reduces jitter
- More stable predictions
- Fewer false positives

## 7. Implementation Details

### 7.1 Software Stack

- **Framework**: PyTorch 2.0+
- **Models**: timm, pytorchvideo
- **Vision**: OpenCV, torchvision
- **Data**: pandas, numpy
- **Metrics**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Logging**: TensorBoard

### 7.2 Hardware Requirements

#### Training
- GPU: NVIDIA RTX 3090 or better (12+ GB VRAM)
- RAM: 32 GB
- Storage: 100+ GB SSD

#### Inference
- GPU: NVIDIA GTX 1660 or better (6+ GB VRAM)
- RAM: 16 GB
- Storage: 10 GB

### 7.3 Computational Complexity

| Model | FLOPs | Parameters | Inference Time (ms) |
|-------|-------|------------|---------------------|
| TimeSformer | ~200 GFLOPs | 87M | 50-70 |
| Video Swin | ~150 GFLOPs | 88M | 40-60 |
| SlowFast | ~100 GFLOPs | 34M | 25-35 |
| Simple3D | ~50 GFLOPs | 15M | 15-20 |

*Measured on RTX 3090 with batch size 1*

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Fixed clip length**: Cannot handle variable-length activities
2. **Single camera**: No multi-camera tracking
3. **Offline training**: No online learning
4. **Limited classes**: Only 3 activity classes
5. **Computational cost**: Requires GPU for real-time inference

### 8.2 Future Improvements

1. **Temporal Action Localization**: Precise start/end times
2. **Multi-camera Fusion**: Track across cameras
3. **Online Learning**: Adapt to new environments
4. **More Classes**: Fine-grained activity recognition
5. **Edge Deployment**: Optimize for edge devices (Jetson, etc.)
6. **Attention Visualization**: Explain model decisions
7. **Active Learning**: Efficient data labeling

## 9. Ethical Considerations

### 9.1 Privacy
- System should comply with privacy regulations
- Anonymize faces and identities
- Secure data storage and transmission

### 9.2 Bias
- Ensure diverse training data
- Test across different demographics
- Monitor for biased predictions

### 9.3 Deployment
- Human-in-the-loop for critical decisions
- Clear alert thresholds
- Regular model audits

## 10. References

1. Bertasius et al. "Is Space-Time Attention All You Need for Video Understanding?" ICML 2021
2. Liu et al. "Video Swin Transformer" CVPR 2022
3. Feichtenhofer et al. "SlowFast Networks for Video Recognition" ICCV 2019
4. Sultani et al. "Real-world Anomaly Detection in Surveillance Videos" CVPR 2018
5. Carreira & Zisserman. "Quo Vadis, Action Recognition?" CVPR 2017
