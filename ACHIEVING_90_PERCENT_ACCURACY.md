# Achieving 90%+ Accuracy with Real Data

## 🎯 Goal: 90%+ Accuracy on Real Theft Detection

To achieve 90%+ accuracy, you need **real surveillance data** and proper training. Here's your complete roadmap:

---

## 📊 Step 1: Get Real Data (CRITICAL)

### Option A: Public Datasets (Recommended)

#### 1. UCF-Crime Dataset
**Best for theft detection**
- **Size**: 1,900 videos, 128 hours
- **Classes**: Robbery, Stealing, Burglary, Shoplifting, etc.
- **Quality**: Real CCTV footage
- **Download**: https://www.crcv.ucf.edu/projects/real-world/

**How to get it:**
```bash
# 1. Visit the website and request access
# 2. Download the dataset (~38 GB)
# 3. Extract to data/raw_videos/
```

**Organize like this:**
```
data/raw_videos/
├── Normal/
│   ├── video001.mp4
│   ├── video002.mp4
│   └── ...
├── Robbery/
│   ├── video001.mp4
│   └── ...
├── Stealing/
│   ├── video001.mp4
│   └── ...
└── Burglary/
    ├── video001.mp4
    └── ...
```

#### 2. Alternative Datasets

**CCTV-Fights Dataset**
- Good for violence detection
- Can be combined with theft data
- Download: https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch

**RWF-2000 Dataset**
- Real-world fight videos
- 2,000 videos
- Good for suspicious activity

### Option B: Collect Your Own Data

If you have access to CCTV footage:

1. **Normal activities**: 500+ clips
   - People walking
   - Standing
   - Talking
   - Shopping normally

2. **Suspicious activities**: 300+ clips
   - Loitering
   - Looking around nervously
   - Following someone
   - Unusual behavior

3. **Theft/Robbery**: 200+ clips
   - Actual theft incidents
   - Shoplifting
   - Pickpocketing
   - Robbery

**Minimum data requirements:**
- **For 90% accuracy**: 1,000+ clips total (300+ per class)
- **For 95% accuracy**: 3,000+ clips total (1,000+ per class)

---

## 🔧 Step 2: Prepare Your Data

### Extract Clips from Videos

```bash
# Extract clips from your organized videos
python scripts/extract_clips.py \
    --input_dir data/raw_videos \
    --output_dir data/clips \
    --clip_len 32 \
    --stride 16 \
    --fps 10 \
    --format npy
```

**This will:**
- Extract 32-frame clips at 10 FPS
- Create overlapping clips (stride=16)
- Generate metadata CSV files
- Split into train/val sets (80/20)

**Expected output:**
```
Extraction completed!
Total clips: 2,450
Train clips: 1,960 -> data/metadata_train.csv
Val clips: 490 -> data/metadata_val.csv

Class distribution:
Normal        820
Suspicious    650
Theft         490
```

---

## 🚀 Step 3: Train High-Accuracy Models

### Model 1: TimeSformer (Highest Accuracy)

**Expected accuracy: 88-92%**

```bash
python src/train.py \
    --train_csv data/metadata_train.csv \
    --val_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --model_name timesformer \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --num_frames 32 \
    --use_class_weights \
    --use_amp \
    --output_dir results/checkpoints/timesformer
```

**Training time**: 4-6 hours on RTX 3090

**Tips for better accuracy:**
- Use `--epochs 50` or more
- Enable `--use_class_weights` for imbalanced data
- Use `--label_smoothing 0.1` for regularization
- Monitor TensorBoard: `tensorboard --logdir results/logs`

### Model 2: SlowFast (Good Balance)

**Expected accuracy: 85-89%**

```bash
python src/train.py \
    --train_csv data/metadata_train.csv \
    --val_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --model_name slowfast \
    --epochs 40 \
    --batch_size 8 \
    --lr 3e-4 \
    --use_class_weights \
    --output_dir results/checkpoints/slowfast
```

**Training time**: 2-3 hours on RTX 3090

### Model 3: Ensemble (Best Accuracy)

**Expected accuracy: 90-94%**

After training both models, create an ensemble:

```bash
# Train TimeSformer
python src/train.py --model_name timesformer --epochs 50 --batch_size 4

# Train SlowFast
python src/train.py --model_name slowfast --epochs 40 --batch_size 8

# Evaluate ensemble
python src/eval.py \
    --test_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --checkpoint results/checkpoints/timesformer/best_acc.pth \
    --model_name timesformer \
    --ensemble \
    --checkpoint2 results/checkpoints/slowfast/best_acc.pth \
    --model_name2 slowfast \
    --output_dir results/eval_ensemble
```

---

## 📈 Step 4: Optimize for 90%+ Accuracy

### Hyperparameter Tuning

**Learning Rate:**
```bash
# Try different learning rates
python src/train.py --model_name timesformer --lr 5e-5  # Lower
python src/train.py --model_name timesformer --lr 2e-4  # Higher
```

**Batch Size:**
```bash
# Larger batch = more stable training
python src/train.py --model_name timesformer --batch_size 8  # If GPU allows
```

**Epochs:**
```bash
# Train longer for better convergence
python src/train.py --model_name timesformer --epochs 60
```

**Data Augmentation:**
The code already includes:
- Random horizontal flip
- Random resized crop
- Color jitter
- Normalization

### Advanced Techniques

**1. Class Weighting (Already Implemented)**
```bash
python src/train.py --use_class_weights  # Handles imbalanced data
```

**2. Label Smoothing (Already Implemented)**
```bash
python src/train.py --label_smoothing 0.1  # Prevents overfitting
```

**3. Mixed Precision Training (Already Implemented)**
```bash
python src/train.py --use_amp  # 2x faster, same accuracy
```

**4. Gradient Clipping (Already Implemented)**
```bash
python src/train.py --grad_clip 1.0  # Prevents exploding gradients
```

---

## 🎯 Step 5: Evaluate and Verify

### Comprehensive Evaluation

```bash
python src/eval.py \
    --test_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --checkpoint results/checkpoints/timesformer/best_acc.pth \
    --model_name timesformer \
    --output_dir results/eval
```

**Expected output:**
```
==================================================
EVALUATION RESULTS
==================================================
Accuracy: 91.30%
F1 Score (Macro): 90.21%
Precision (Macro): 91.45%
Recall (Macro): 89.01%
ROC-AUC: 95.50%

Per-class metrics:
  normal:
    Precision: 94.50%
    Recall: 96.00%
    F1-Score: 95.24%
  suspicious:
    Precision: 88.40%
    Recall: 85.20%
    F1-Score: 86.77%
  theft:
    Precision: 91.45%
    Recall: 85.80%
    F1-Score: 88.53%
```

### Check Results

1. **Confusion Matrix**: `results/figures/timesformer_confusion_matrix.png`
2. **ROC Curve**: `results/figures/timesformer_roc_curve.png`
3. **Training Curves**: `results/figures/timesformer_training_curves.png`
4. **Metrics JSON**: `results/metrics/timesformer_metrics.json`

---

## 📊 Expected Performance by Dataset Size

| Dataset Size | Model | Expected Accuracy |
|--------------|-------|-------------------|
| 300 clips | Simple3D | 70-75% |
| 500 clips | Simple3D | 75-80% |
| 1,000 clips | TimeSformer | 82-87% |
| 2,000 clips | TimeSformer | 87-91% |
| 3,000+ clips | TimeSformer | 90-93% |
| 3,000+ clips | Ensemble | 92-95% |

---

## 🔍 Troubleshooting Low Accuracy

### If accuracy is below 90%:

**1. Check Data Quality**
```bash
# Preview your data
jupyter notebook notebooks/00_data_preview.ipynb
```

**Issues to look for:**
- Mislabeled videos
- Poor video quality
- Class imbalance
- Too short clips

**2. Check Training Progress**
```bash
tensorboard --logdir results/logs
```

**Look for:**
- Loss decreasing steadily
- No overfitting (train/val gap)
- Accuracy improving

**3. Try Different Models**
```bash
# If TimeSformer is too slow
python src/train.py --model_name slowfast --epochs 40

# If need more speed
python src/train.py --model_name simple3d --epochs 30
```

**4. Collect More Data**
- Minimum 1,000 clips for 90% accuracy
- Balanced classes (similar number per class)
- High-quality videos

**5. Fine-tune Hyperparameters**
```bash
# Lower learning rate
python src/train.py --lr 5e-5 --epochs 60

# Increase batch size
python src/train.py --batch_size 8

# More regularization
python src/train.py --dropout 0.4 --label_smoothing 0.15
```

---

## 📋 Complete Workflow for 90%+ Accuracy

### Week 1: Data Collection
```bash
# Day 1-3: Download UCF-Crime dataset
# Day 4-5: Organize videos by class
# Day 6-7: Extract clips and verify data
python scripts/extract_clips.py --input_dir data/raw_videos --output_dir data/clips
```

### Week 2: Initial Training
```bash
# Day 1-2: Train Simple3D (baseline)
python src/train.py --model_name simple3d --epochs 20

# Day 3-5: Train TimeSformer
python src/train.py --model_name timesformer --epochs 50 --batch_size 4

# Day 6-7: Evaluate and analyze
python src/eval.py --checkpoint results/checkpoints/timesformer/best_acc.pth --model_name timesformer
```

### Week 3: Optimization
```bash
# Day 1-3: Train SlowFast
python src/train.py --model_name slowfast --epochs 40

# Day 4-5: Create ensemble
python src/eval.py --ensemble --checkpoint results/checkpoints/timesformer/best_acc.pth --checkpoint2 results/checkpoints/slowfast/best_acc.pth

# Day 6-7: Fine-tune best model
python src/train.py --model_name timesformer --epochs 60 --lr 5e-5
```

### Week 4: Testing & Deployment
```bash
# Day 1-2: Test on real videos
python src/infer_video.py --video test_video.mp4 --checkpoint results/checkpoints/timesformer/best_acc.pth

# Day 3-4: Real-time testing
python src/realtime_webcam.py --checkpoint results/checkpoints/timesformer/best_acc.pth

# Day 5-7: Documentation and presentation
```

---

## 🎓 Tips from Research

### What Works Best:

1. **Large Dataset**: 2,000+ clips minimum
2. **Pretrained Models**: Use Kinetics-400 pretrained weights
3. **Ensemble**: Combine transformer + 3D CNN
4. **Data Augmentation**: Spatial + temporal
5. **Class Balancing**: Use class weights
6. **Long Training**: 50+ epochs for transformers
7. **Mixed Precision**: Faster training, same accuracy

### What Doesn't Work:

1. ❌ Training from scratch (no pretrained weights)
2. ❌ Too small dataset (<500 clips)
3. ❌ Single model without ensemble
4. ❌ No data augmentation
5. ❌ Ignoring class imbalance
6. ❌ Too few epochs (<20)
7. ❌ Wrong learning rate

---

## 📞 Need Help?

### If you're stuck:

1. **Check FAQ.md** - 50 common questions answered
2. **Review training logs** - Use TensorBoard
3. **Verify data quality** - Use data preview notebook
4. **Try different models** - Start with Simple3D, then TimeSformer
5. **Collect more data** - More data = better accuracy

### Quick Checklist:

- [ ] Have 1,000+ real video clips
- [ ] Organized by class (normal/suspicious/theft)
- [ ] Extracted clips with proper metadata
- [ ] Trained TimeSformer for 50+ epochs
- [ ] Used pretrained weights
- [ ] Enabled class weighting
- [ ] Evaluated on separate test set
- [ ] Checked confusion matrix
- [ ] Tried ensemble if needed

---

## 🎯 Summary

**To achieve 90%+ accuracy:**

1. ✅ Get **real data** (UCF-Crime or similar) - 2,000+ clips
2. ✅ Use **TimeSformer** model with pretrained weights
3. ✅ Train for **50+ epochs** with proper hyperparameters
4. ✅ Enable **class weighting** and **data augmentation**
5. ✅ Create **ensemble** of TimeSformer + SlowFast
6. ✅ Evaluate thoroughly and iterate

**Expected timeline**: 3-4 weeks from data collection to 90%+ accuracy

**Hardware needed**: NVIDIA GPU with 12+ GB VRAM (RTX 3060 Ti or better)

---

**Good luck achieving 90%+ accuracy! 🚀**
