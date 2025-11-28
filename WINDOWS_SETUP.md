# Windows Setup Guide

## Common Windows Issues & Fixes

### Issue 1: DataLoader Memory Error ✅ FIXED

**Error**: `"The paging file is too small"` or `MemoryError` with DataLoader

**Solution**: The code now automatically detects Windows and sets `num_workers=0` to disable multiprocessing.

If you still have issues, you can manually set it:
```bash
python src/train.py --num_workers 0 --model_name simple3d --epochs 10
```

### Issue 2: CUDA Out of Memory

**Solution**: Reduce batch size
```bash
python src/train.py --batch_size 2 --model_name simple3d --epochs 10
```

### Issue 3: Slow Training on Windows

**Tips**:
1. Use smaller models first: `--model_name simple3d`
2. Reduce batch size: `--batch_size 4`
3. Use fewer frames: `--num_frames 16`
4. Disable AMP if issues: Remove `--use_amp` flag

## Recommended Windows Training Command

```bash
# Quick demo (works on most Windows systems)
python src/train.py ^
    --model_name simple3d ^
    --epochs 10 ^
    --batch_size 4 ^
    --num_workers 0 ^
    --lr 3e-4
```

## Windows-Specific Tips

### 1. Virtual Environment
```powershell
# Create venv
python -m venv venv

# Activate (PowerShell)
venv\Scripts\Activate.ps1

# Activate (CMD)
venv\Scripts\activate.bat
```

### 2. CUDA Installation
If you have NVIDIA GPU:
```powershell
# Check CUDA
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Memory Management
- Close other applications
- Increase virtual memory (paging file)
- Use Task Manager to monitor memory usage

### 4. Path Issues
Always use forward slashes or raw strings:
```python
# Good
path = "data/clips"
path = r"data\clips"

# Avoid
path = "data\clips"  # May cause issues
```

## Quick Test

Run this to verify everything works:
```bash
python test_installation.py
```

## Training on Windows (Step-by-Step)

### Step 1: Create Data
```bash
python -c "from src.datasets import create_dummy_dataset; create_dummy_dataset()"
```

### Step 2: Train (Windows-optimized)
```bash
python src/train.py --model_name simple3d --epochs 10 --batch_size 4
```

### Step 3: Evaluate
```bash
python src/eval.py --checkpoint results/checkpoints/best_acc.pth --model_name simple3d
```

## Performance on Windows

Expected performance (RTX 3060):
- Simple3D: 40-50 FPS
- SlowFast: 25-30 FPS
- TimeSformer: 12-15 FPS

## Alternative: Use Google Colab

If Windows issues persist, use Google Colab (free GPU):
1. Open `notebooks/01_quick_train_colab.ipynb`
2. Upload to Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run all cells

## Common Commands (Windows)

```powershell
# Create dummy data
python -c "from src.datasets import create_dummy_dataset; create_dummy_dataset()"

# Train simple model
python src/train.py --model_name simple3d --epochs 10

# Train with small batch
python src/train.py --model_name simple3d --epochs 10 --batch_size 2

# Evaluate
python src/eval.py --checkpoint results/checkpoints/best_acc.pth --model_name simple3d

# Infer video
python src/infer_video.py --video test.mp4 --checkpoint results/checkpoints/best_acc.pth --model_name simple3d

# Webcam (may be slow on Windows without GPU)
python src/realtime_webcam.py --checkpoint results/checkpoints/best_acc.pth --model_name simple3d
```

## Troubleshooting Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] CUDA available (if using GPU): `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Enough disk space (20+ GB)
- [ ] Enough RAM (16+ GB)
- [ ] Using `--num_workers 0` on Windows
- [ ] Batch size appropriate for your GPU

## Getting Help

If issues persist:
1. Check error message carefully
2. Review `FAQ.md`
3. Try Google Colab instead
4. Reduce batch size and num_workers
5. Use Simple3D model first

---

**The code is now optimized for Windows. Try the training command again!**
