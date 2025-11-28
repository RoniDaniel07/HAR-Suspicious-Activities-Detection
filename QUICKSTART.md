# Quick Start Guide - HAR Theft Detection

Get up and running in 5 minutes!

## 🚀 Super Quick Demo (Windows)

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create dummy dataset
python -c "from src.datasets import create_dummy_dataset; create_dummy_dataset()"

# 3. Train a quick model (5-10 minutes)
python src/train.py --model_name simple3d --epochs 10 --batch_size 8

# 4. Test inference
python src/infer_video.py --video test.mp4 --checkpoint results/checkpoints/best_acc.pth --model_name simple3d
```

## 📋 Step-by-Step Guide

### Step 1: Environment Setup

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Step 2: Verify Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
```

### Step 3: Create Demo Dataset

```powershell
python -c "from src.datasets import create_dummy_dataset; create_dummy_dataset(num_videos_per_class=10)"
```

This creates:
- `data/metadata_train.csv` (24 clips)
- `data/metadata_val.csv` (6 clips)
- `data/clips/` (30 .npy files)

### Step 4: Train Your First Model

**Option A: Fast Training (Simple3D - 5 minutes)**
```powershell
python src/train.py ^
    --model_name simple3d ^
    --epochs 10 ^
    --batch_size 8 ^
    --lr 3e-4
```

**Option B: High Accuracy (TimeSformer - 15 minutes)**
```powershell
python src/train.py ^
    --model_name timesformer ^
    --epochs 15 ^
    --batch_size 2 ^
    --lr 1e-4
```

### Step 5: View Results

Training outputs:
- Checkpoints: `results/checkpoints/best_acc.pth`
- Metrics: `results/metrics/`
- Plots: `results/figures/`
- Logs: `results/logs/` (view with TensorBoard)

View TensorBoard:
```powershell
tensorboard --logdir results/logs
```

### Step 6: Run Inference

**On a video file:**
```powershell
python src/infer_video.py ^
    --video input.mp4 ^
    --checkpoint results/checkpoints/best_acc.pth ^
    --model_name simple3d ^
    --output_video output.mp4 ^
    --output_csv detections.csv
```

**On webcam (real-time):**
```powershell
python src/realtime_webcam.py ^
    --checkpoint results/checkpoints/best_acc.pth ^
    --model_name simple3d
```

## 🎓 Google Colab (No Local Setup Required)

1. Open `notebooks/01_quick_train_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells
4. Download results

## 📊 Expected Results (Dummy Dataset)

Since the dummy dataset is random, expect:
- Accuracy: 40-60% (random baseline ~33%)
- Training time: 5-10 minutes
- The model will learn some patterns even from random data

**For real results**, use actual surveillance videos!

## 🔧 Troubleshooting

### CUDA Out of Memory
```powershell
# Reduce batch size
python src/train.py --model_name simple3d --batch_size 2
```

### Slow Training
```powershell
# Enable mixed precision
python src/train.py --model_name simple3d --use_amp
```

### Import Errors
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### No GPU Available
```powershell
# Train on CPU (slower)
python src/train.py --model_name simple3d --device cpu --batch_size 2
```

## 📚 Next Steps

1. **Get Real Data**: Download UCF-Crime or collect your own videos
2. **Extract Clips**: Use `scripts/extract_clips.py`
3. **Train Better Models**: Try TimeSformer or ensemble
4. **Optimize**: Tune hyperparameters
5. **Deploy**: Set up real-time monitoring

## 🎯 Quick Commands Reference

```powershell
# Create dummy data
python -c "from src.datasets import create_dummy_dataset; create_dummy_dataset()"

# Train simple model
python src/train.py --model_name simple3d --epochs 10

# Train transformer
python src/train.py --model_name timesformer --epochs 25 --batch_size 2

# Evaluate
python src/eval.py --checkpoint results/checkpoints/best_acc.pth --model_name simple3d

# Infer video
python src/infer_video.py --video test.mp4 --checkpoint results/checkpoints/best_acc.pth --model_name simple3d

# Real-time webcam
python src/realtime_webcam.py --checkpoint results/checkpoints/best_acc.pth --model_name simple3d

# Extract clips from videos
python scripts/extract_clips.py --input_dir data/raw_videos --output_dir data/clips
```

## 💡 Tips

1. **Start Small**: Use dummy data or small subset first
2. **Monitor Training**: Use TensorBoard to watch progress
3. **Save Checkpoints**: Models are saved automatically
4. **Try Ensemble**: Combine models for best accuracy
5. **Adjust Threshold**: Tune detection threshold for your use case

## 📞 Need Help?

- Check `README.md` for detailed documentation
- See `report/methodology.md` for technical details
- Open an issue on GitHub
- Review example notebooks in `notebooks/`

---

**Ready to detect theft? Let's go! 🚀**
