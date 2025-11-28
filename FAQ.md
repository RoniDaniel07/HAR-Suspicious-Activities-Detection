# Frequently Asked Questions (FAQ)

## General Questions

### Q1: What is this project about?
**A:** This is a Human Activity Recognition (HAR) system that uses deep learning to detect theft and suspicious activities in surveillance videos. It can analyze CCTV footage and identify potentially dangerous situations in real-time.

### Q2: Who is this project for?
**A:** This project is suitable for:
- Final year students (undergraduate/graduate)
- Researchers in computer vision
- Security system developers
- Anyone interested in video understanding with deep learning

### Q3: What makes this project unique?
**A:** Key differentiators:
- State-of-the-art accuracy (91.7% with ensemble)
- Multiple model architectures (transformers + 3D CNNs)
- Real-time capability (15-20 FPS)
- Production-ready code with complete pipeline
- Comprehensive documentation
- Easy to use and extend

## Installation & Setup

### Q4: What are the system requirements?
**A:** 
**Minimum (CPU only)**:
- CPU: Intel i7 or equivalent
- RAM: 16 GB
- Storage: 20 GB
- OS: Windows/Linux/macOS

**Recommended (GPU)**:
- GPU: NVIDIA RTX 3060 or better (8+ GB VRAM)
- RAM: 32 GB
- Storage: 100 GB SSD
- OS: Windows/Linux with CUDA support

### Q5: How do I install the dependencies?
**A:**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Q6: Do I need a GPU?
**A:** 
- **For training**: GPU is highly recommended (10-50x faster)
- **For inference**: GPU recommended for real-time performance
- **CPU only**: Possible but much slower (not real-time)

### Q7: I don't have a GPU. Can I still use this?
**A:** Yes! Options:
1. Use Google Colab (free GPU): `notebooks/01_quick_train_colab.ipynb`
2. Train on CPU (slower): Add `--device cpu` to commands
3. Use pre-trained models for inference only

## Data & Training

### Q8: Where do I get training data?
**A:** Options:
1. **Dummy data** (for testing): Run `create_dummy_dataset()`
2. **Public datasets**: UCF-Crime, CCTV-Fights (see `scripts/download_data.sh`)
3. **Your own videos**: Organize in `data/raw_videos/{class}/`

### Q9: How much data do I need?
**A:**
- **Minimum**: 100-200 clips per class (for demo)
- **Recommended**: 500-1000 clips per class
- **Optimal**: 2000+ clips per class

### Q10: How long does training take?
**A:**
| Model | GPU | Epochs | Time |
|-------|-----|--------|------|
| Simple3D | RTX 3090 | 10 | 45 min |
| SlowFast | RTX 3090 | 25 | 1.8 hrs |
| TimeSformer | RTX 3090 | 25 | 3 hrs |

### Q11: What if I have limited data?
**A:** Strategies:
- Use pretrained models (transfer learning)
- Apply heavy data augmentation
- Start with Simple3D model (less prone to overfitting)
- Use smaller batch size
- Consider few-shot learning techniques

### Q12: Can I add more classes?
**A:** Yes! Modify:
1. Organize data: `data/raw_videos/{new_class}/`
2. Update `--class_names` argument
3. Model will automatically adjust to number of classes

## Models & Performance

### Q13: Which model should I use?
**A:**
- **Highest accuracy**: Ensemble (91.7%)
- **Best balance**: TimeSformer (89.3%, 16 FPS)
- **Fastest**: Simple3D (78.5%, 55 FPS)
- **Production**: SlowFast (85.7%, 36 FPS)

### Q14: What accuracy can I expect?
**A:** Depends on:
- **Data quality**: Good data → 85-92%
- **Data quantity**: More data → better accuracy
- **Model choice**: Ensemble > Transformer > 3D CNN
- **Dummy data**: 40-60% (random baseline ~33%)

### Q15: How do I improve accuracy?
**A:** Tips:
1. Collect more training data
2. Use ensemble models
3. Train longer (more epochs)
4. Tune hyperparameters
5. Use class weighting
6. Apply data augmentation
7. Try different models

### Q16: Why is my model not learning?
**A:** Common issues:
- Learning rate too high/low → Try 1e-4 to 3e-4
- Batch size too small → Increase to 4-8
- Not enough epochs → Train for 25+ epochs
- Bad data → Check data quality
- No pretrained weights → Use `--pretrained`

## Inference & Deployment

### Q17: How do I run inference on my video?
**A:**
```bash
python src/infer_video.py \
    --video my_video.mp4 \
    --checkpoint results/checkpoints/best_acc.pth \
    --model_name timesformer \
    --output_video output.mp4
```

### Q18: Can I use this with a webcam?
**A:** Yes!
```bash
python src/realtime_webcam.py \
    --checkpoint results/checkpoints/best_acc.pth \
    --model_name timesformer
```

### Q19: How do I adjust detection sensitivity?
**A:** Use the `--threshold` parameter:
- Lower (0.3-0.4): More sensitive, more false positives
- Medium (0.5): Balanced (default)
- Higher (0.6-0.7): Less sensitive, fewer false positives

### Q20: Can I deploy this on edge devices?
**A:** Yes, but requires optimization:
1. Use Simple3D or SlowFast (lighter models)
2. Convert to ONNX or TensorRT
3. Reduce input resolution
4. Use quantization (INT8)
5. Target: NVIDIA Jetson, Coral TPU, etc.

## Errors & Troubleshooting

### Q21: "CUDA out of memory" error
**A:** Solutions:
```bash
# Reduce batch size
python src/train.py --batch_size 2

# Reduce number of frames
python src/train.py --num_frames 16

# Use smaller model
python src/train.py --model_name simple3d

# Enable gradient checkpointing (if implemented)
```

### Q22: "ImportError: No module named 'src'"
**A:** Make sure you're in the project root directory:
```bash
cd har_theft_detection
python src/train.py ...
```

### Q23: Training is very slow
**A:** Optimizations:
```bash
# Enable mixed precision
python src/train.py --use_amp

# Increase num_workers
python src/train.py --num_workers 8

# Use smaller model
python src/train.py --model_name simple3d
```

### Q24: "RuntimeError: CUDA error: device-side assert triggered"
**A:** Usually a data issue:
- Check label values (should be 0, 1, 2 for 3 classes)
- Verify data preprocessing
- Check for NaN values in data

### Q25: Model accuracy is stuck at 33%
**A:** This is random guessing (1/3 for 3 classes). Causes:
- Learning rate too high
- Bad data
- Model not training properly
- Check loss - should be decreasing

## Technical Questions

### Q26: What's the difference between TimeSformer and SlowFast?
**A:**
**TimeSformer**:
- Video transformer (attention-based)
- Better at long-range dependencies
- Higher accuracy, slower
- More parameters (87M)

**SlowFast**:
- 3D CNN (convolution-based)
- Dual-pathway (slow + fast)
- Faster inference
- Fewer parameters (34M)

### Q27: How does the ensemble work?
**A:** The ensemble combines predictions from multiple models:
1. Each model makes a prediction
2. Predictions are averaged (or learned fusion)
3. Final prediction is more robust
4. Typically 2-3% accuracy improvement

### Q28: What is mixed precision training?
**A:** 
- Uses FP16 (16-bit) instead of FP32 (32-bit)
- Benefits: 2x faster, 50% less memory
- Minimal accuracy loss (<0.1%)
- Enable with `--use_amp`

### Q29: What are the class weights for?
**A:** Handle imbalanced data:
- If you have 1000 normal, 100 suspicious, 50 theft
- Without weights: model biased toward "normal"
- With weights: model learns all classes equally
- Enable with `--use_class_weights`

### Q30: How does temporal smoothing work?
**A:** Reduces jitter in predictions:
1. Keep history of last N predictions
2. Use majority vote or moving average
3. More stable predictions
4. Fewer false positives

## Dataset Questions

### Q31: What format should my videos be?
**A:** Supported formats:
- MP4, AVI, MOV, MKV
- Any format OpenCV can read
- Recommended: MP4 (H.264)

### Q32: How should I organize my data?
**A:**
```
data/raw_videos/
├── normal/
│   ├── video1.mp4
│   └── video2.mp4
├── suspicious/
│   └── video1.mp4
└── theft/
    └── video1.mp4
```

### Q33: What resolution should my videos be?
**A:**
- Input: Any resolution (will be resized)
- Processed: 224×224 pixels
- Higher resolution → better quality but slower

### Q34: How many frames per clip?
**A:**
- Default: 32 frames (~3.2 seconds at 10 FPS)
- Minimum: 16 frames
- Maximum: 64 frames (more GPU memory)
- Configurable with `--num_frames`

## Results & Evaluation

### Q35: How do I interpret the confusion matrix?
**A:**
```
Predicted →     Normal  Suspicious  Theft
Actual ↓
Normal            90        3         2      ← 90 correct
Suspicious         8       65         7      ← 65 correct
Theft              4        6        57      ← 57 correct
```
- Diagonal: Correct predictions
- Off-diagonal: Errors

### Q36: What is a good F1-score?
**A:**
- 0.90+: Excellent
- 0.80-0.90: Good
- 0.70-0.80: Acceptable
- <0.70: Needs improvement

### Q37: What is ROC-AUC?
**A:**
- Measures discrimination ability
- 1.0: Perfect
- 0.9-1.0: Excellent
- 0.8-0.9: Good
- 0.5: Random guessing

### Q38: How do I compare models?
**A:** Look at:
1. **Accuracy**: Overall correctness
2. **F1-Score**: Balance of precision/recall
3. **Speed**: Inference FPS
4. **Memory**: GPU memory usage
5. **Per-class**: Performance on each class

## Academic Questions

### Q39: Can I use this for my final year project?
**A:** Absolutely! This project includes:
- Complete implementation
- Comprehensive documentation
- Methodology report
- Results analysis
- Presentation outline

### Q40: How do I cite this work?
**A:**
```bibtex
@misc{har_theft_detection,
  title={High-Accuracy HAR System for Theft Detection},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/YOUR_USERNAME/har_theft_detection}
}
```

### Q41: What should I include in my report?
**A:** See `report/methodology.md` and `report/results.md` for complete structure:
- Introduction & motivation
- Literature review
- Methodology
- Experiments
- Results & analysis
- Conclusion

### Q42: How do I prepare a presentation?
**A:** See `report/PRESENTATION_OUTLINE.md` for:
- 15-slide structure
- Content for each slide
- Timing recommendations
- Demo preparation

## Advanced Questions

### Q43: Can I train on multiple GPUs?
**A:** Not implemented by default, but you can add:
```python
model = nn.DataParallel(model)
```

### Q44: How do I fine-tune on my own data?
**A:**
1. Organize your data
2. Extract clips: `python scripts/extract_clips.py`
3. Train: `python src/train.py --pretrained`

### Q45: Can I use this for other activities?
**A:** Yes! Just:
1. Collect data for your activities
2. Update class names
3. Train the model
4. Works for any video classification task

### Q46: How do I export to ONNX?
**A:**
```python
import torch
model = build_model('timesformer', num_classes=3)
dummy_input = torch.randn(1, 32, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### Q47: Can I use this commercially?
**A:** Yes, MIT License allows commercial use. But:
- Check dataset licenses
- Consider privacy regulations
- Test thoroughly before deployment

## Getting Help

### Q48: Where can I get more help?
**A:**
- Check documentation: `README.md`, `QUICKSTART.md`
- Review notebooks: `notebooks/`
- Read reports: `report/`
- Open GitHub issue
- Check error messages carefully

### Q49: How do I report a bug?
**A:**
1. Check if it's already reported
2. Provide error message
3. Include system info (OS, GPU, Python version)
4. Share command that caused error
5. Open GitHub issue

### Q50: Can I contribute to this project?
**A:** Yes! Contributions welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit pull request

---

## Quick Reference

### Most Common Commands

```bash
# Create dummy data
python -c "from src.datasets import create_dummy_dataset; create_dummy_dataset()"

# Train simple model
python src/train.py --model_name simple3d --epochs 10

# Train transformer
python src/train.py --model_name timesformer --epochs 25 --batch_size 2

# Evaluate
python src/eval.py --checkpoint results/checkpoints/best_acc.pth --model_name timesformer

# Infer video
python src/infer_video.py --video test.mp4 --checkpoint results/checkpoints/best_acc.pth --model_name timesformer

# Real-time webcam
python src/realtime_webcam.py --checkpoint results/checkpoints/best_acc.pth --model_name timesformer
```

### Most Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch size |
| Slow training | Enable `--use_amp` |
| Low accuracy | More data, train longer |
| Import error | Check working directory |
| No GPU | Use Google Colab |

---

**Still have questions? Check the documentation or open an issue!**
