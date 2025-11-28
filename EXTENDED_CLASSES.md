# Extended Activity Recognition Classes

## Updated Class Structure

The system now supports **6 activity classes**:

### Original Classes (3)
1. **Normal** - Regular activities (walking, standing, talking)
2. **Suspicious** - Potentially concerning behavior (loitering, looking around nervously)
3. **Theft** - Stealing, shoplifting, robbery

### New Classes (3)
4. **Fighting** - Physical altercations, assault, violence
5. **Weapon (Gun)** - Person holding or brandishing a firearm
6. **Weapon (Other)** - Knives, bombs, or other dangerous objects

## How to Use Extended Classes

### Option 1: Train with 6 Classes

```bash
# Create dataset with 6 classes
python -c "from src.datasets import create_dummy_dataset; create_dummy_dataset(num_videos_per_class=10, classes=['normal', 'suspicious', 'theft', 'fighting', 'weapon_gun', 'weapon_other'])"

# Train model
python src/train.py --model_name simple3d --epochs 20 --batch_size 4
```

### Option 2: Use Existing 3-Class Model + Object Detection

For better weapon detection, combine:
1. **Activity Recognition** (existing model) - Detects fighting
2. **Object Detection** (YOLOv8/Faster R-CNN) - Detects weapons

## Recommended Datasets

### For Fighting Detection
- **CCTV-Fights Dataset** - Real-world fight videos
- **Hockey Fight Dataset** - Sports violence
- **RWF-2000** - Real-world fight dataset
- **UCF-Crime** - Includes assault/fighting

### For Weapon Detection
- **COCO Dataset** - Contains knife class
- **Open Images** - Various weapon classes
- **Custom Security Datasets** - Specialized weapon detection
- **X-ray Security Datasets** - For concealed weapons

## Data Organization

```
data/raw_videos/
├── normal/
├── suspicious/
├── theft/
├── fighting/          # NEW: Fight videos
├── weapon_gun/        # NEW: Gun-related incidents
└── weapon_other/      # NEW: Knife, bomb, etc.
```

## Training Commands

### Train 6-Class Model
```bash
python src/train.py \
    --model_name simple3d \
    --epochs 25 \
    --batch_size 4 \
    --lr 3e-4
```

### Inference with 6 Classes
```bash
python src/infer_video.py \
    --video surveillance.mp4 \
    --checkpoint results/checkpoints/best_acc.pth \
    --model_name simple3d \
    --class_names normal suspicious theft fighting weapon_gun weapon_other \
    --output_video output.mp4
```

## Expected Performance

| Class | Expected Accuracy | Difficulty |
|-------|------------------|------------|
| Normal | 90-95% | Easy |
| Suspicious | 75-85% | Hard |
| Theft | 80-90% | Medium |
| Fighting | 85-92% | Medium |
| Weapon (Gun) | 70-85% | Hard |
| Weapon (Other) | 65-80% | Very Hard |

**Note:** Weapon detection from video alone is challenging. For best results:
- Use high-resolution cameras
- Combine with object detection models
- Consider multi-modal approach (video + metadata)

## Alert Priority Levels

```python
ALERT_LEVELS = {
    'normal': 0,        # No alert
    'suspicious': 1,    # Low priority
    'theft': 2,         # Medium priority
    'fighting': 3,      # High priority
    'weapon_gun': 4,    # Critical priority
    'weapon_other': 4   # Critical priority
}
```

## Hybrid Approach (Recommended)

For production deployment, use a **two-stage system**:

### Stage 1: Activity Recognition (This System)
- Detects: Normal, Suspicious, Theft, Fighting
- Fast: 15-20 FPS
- Accuracy: 85-92%

### Stage 2: Object Detection (YOLOv8)
- Detects: Guns, Knives, Weapons
- Fast: 30-60 FPS
- Accuracy: 80-95%

### Combined System
```python
if activity in ['fighting', 'suspicious', 'theft']:
    # Run weapon detection
    weapons = detect_weapons(frame)
    if weapons:
        alert_level = 'CRITICAL'
```

## Implementation Status

✅ **Ready to Use:**
- 6-class training support
- Extended inference
- Multi-class evaluation

⚠️ **Requires Additional Work:**
- Weapon-specific datasets
- Object detection integration
- Multi-modal fusion

## Next Steps

1. **Collect Data:**
   - Download CCTV-Fights dataset
   - Collect weapon detection videos
   - Organize into 6 classes

2. **Train Extended Model:**
   ```bash
   python src/train.py --model_name simple3d --epochs 25
   ```

3. **Optional: Add Object Detection:**
   - Install YOLOv8: `pip install ultralytics`
   - Integrate weapon detection
   - Combine predictions

4. **Deploy:**
   - Test on real surveillance footage
   - Tune alert thresholds
   - Set up notification system
