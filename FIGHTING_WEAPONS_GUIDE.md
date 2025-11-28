# Fighting & Weapons Detection Guide

## Quick Start

### Option 1: Extended 6-Class System (Recommended)

```bash
# Step 1: Create extended dataset
python create_extended_dataset.py

# Step 2: Train model
python src/train.py --model_name simple3d --epochs 20 --batch_size 4

# Step 3: Run inference
python src/infer_video.py \
    --video surveillance.mp4 \
    --checkpoint results/checkpoints/best_acc.pth \
    --model_name simple3d \
    --class_names normal suspicious theft fighting weapon_gun weapon_other \
    --output_video output.mp4
```

### Option 2: Keep 3 Classes + Add Object Detection

Use existing 3-class model for activities, add YOLOv8 for weapons:

```bash
# Install YOLOv8
pip install ultralytics

# Use hybrid detection (see below)
```

## Class Definitions

| Class | Description | Examples | Alert Level |
|-------|-------------|----------|-------------|
| **Normal** | Regular activities | Walking, standing, talking | None |
| **Suspicious** | Concerning behavior | Loitering, nervous behavior | Low |
| **Theft** | Stealing | Shoplifting, robbery | Medium |
| **Fighting** | Physical violence | Assault, brawl, attack | High |
| **Weapon (Gun)** | Firearms | Handgun, rifle visible | Critical |
| **Weapon (Other)** | Other weapons | Knife, bomb, bat | Critical |

## Data Collection

### Where to Get Data

#### Fighting Videos
1. **CCTV-Fights Dataset**
   - 1,000 fight videos from surveillance cameras
   - Download: Search "CCTV-Fights dataset GitHub"

2. **RWF-2000 Dataset**
   - 2,000 videos (1,000 fights, 1,000 non-fights)
   - Real-world surveillance footage

3. **Hockey Fight Dataset**
   - Sports violence detection
   - Good for training fight recognition

#### Weapon Videos
1. **Custom Collection**
   - Security training videos
   - News footage (with permission)
   - Simulated scenarios

2. **Public Datasets**
   - COCO (contains knife class)
   - Open Images (weapon categories)
   - YouTube (with proper licensing)

### Organize Your Data

```
data/raw_videos/
├── normal/
│   ├── video1.mp4
│   └── video2.mp4
├── suspicious/
│   └── video1.mp4
├── theft/
│   └── video1.mp4
├── fighting/           # Add fight videos here
│   ├── fight1.mp4
│   ├── fight2.mp4
│   └── fight3.mp4
├── weapon_gun/         # Add gun videos here
│   ├── gun1.mp4
│   └── gun2.mp4
└── weapon_other/       # Add knife/bomb videos here
    ├── knife1.mp4
    └── bomb1.mp4
```

## Training Process

### 1. Extract Clips

```bash
python scripts/extract_clips.py \
    --input_dir data/raw_videos \
    --output_dir data/clips \
    --clip_len 32 \
    --fps 10
```

### 2. Train Model

```bash
# For 6 classes
python src/train.py \
    --model_name simple3d \
    --epochs 25 \
    --batch_size 4 \
    --lr 3e-4

# For better accuracy (slower)
python src/train.py \
    --model_name timesformer \
    --epochs 30 \
    --batch_size 2 \
    --lr 1e-4
```

### 3. Evaluate

```bash
python src/eval.py \
    --checkpoint results/checkpoints/best_acc.pth \
    --model_name simple3d \
    --test_csv data/metadata_val.csv
```

## Hybrid System (Activity + Object Detection)

For best weapon detection, combine activity recognition with object detection:

### Install YOLOv8

```bash
pip install ultralytics
```

### Create Hybrid Detection Script

```python
# hybrid_detection.py
from ultralytics import YOLO
import cv2

# Load models
activity_model = load_activity_model()  # Your trained model
weapon_detector = YOLO('yolov8n.pt')    # YOLOv8 for objects

# Process video
for frame in video:
    # Detect activity
    activity = activity_model.predict(frame)
    
    # If suspicious/fighting, check for weapons
    if activity in ['suspicious', 'fighting', 'theft']:
        weapons = weapon_detector.predict(frame, classes=['knife', 'gun'])
        
        if weapons:
            alert = "CRITICAL: Weapon detected!"
        else:
            alert = f"WARNING: {activity} detected"
```

## Alert System

### Priority Levels

```python
ALERT_PRIORITY = {
    'normal': 0,           # No alert
    'suspicious': 1,       # Monitor
    'theft': 2,            # Alert security
    'fighting': 3,         # Immediate response
    'weapon_gun': 4,       # Emergency - call police
    'weapon_other': 4      # Emergency - call police
}
```

### Notification Example

```python
def send_alert(detection):
    priority = ALERT_PRIORITY[detection['class']]
    
    if priority >= 4:
        # Critical: Call police, lock doors
        call_emergency_services()
        trigger_lockdown()
        
    elif priority >= 3:
        # High: Alert security, sound alarm
        notify_security_team()
        sound_alarm()
        
    elif priority >= 2:
        # Medium: Log and notify
        log_incident()
        notify_supervisor()
```

## Performance Expectations

### Activity Recognition (Video-based)

| Class | Accuracy | Speed | Difficulty |
|-------|----------|-------|------------|
| Normal | 90-95% | Fast | Easy |
| Suspicious | 75-85% | Fast | Hard |
| Theft | 80-90% | Fast | Medium |
| Fighting | 85-92% | Fast | Medium |

### Weapon Detection (Object-based)

| Method | Accuracy | Speed | Notes |
|--------|----------|-------|-------|
| Video only | 65-80% | Fast | Limited by resolution |
| YOLOv8 | 85-95% | Very Fast | Needs clear view |
| Hybrid | 90-97% | Fast | Best approach |

## Challenges & Solutions

### Challenge 1: Weapon Detection from Video
**Problem:** Weapons are small, often concealed
**Solution:** 
- Use high-resolution cameras (1080p+)
- Combine with object detection
- Focus on behavior patterns

### Challenge 2: False Positives
**Problem:** Normal objects mistaken for weapons
**Solution:**
- Use temporal smoothing
- Require multiple frame confirmations
- Set appropriate thresholds

### Challenge 3: Occlusion
**Problem:** Weapons/people partially hidden
**Solution:**
- Multi-camera setup
- Track across frames
- Use context (fighting + object = likely weapon)

## Best Practices

### 1. Camera Placement
- ✅ High resolution (1080p minimum)
- ✅ Good lighting
- ✅ Multiple angles
- ✅ Cover entry/exit points

### 2. Model Configuration
- ✅ Use ensemble for critical areas
- ✅ Lower threshold for weapons (more sensitive)
- ✅ Higher threshold for suspicious (fewer false alarms)

### 3. Alert Management
- ✅ Immediate alerts for weapons
- ✅ Logged alerts for suspicious
- ✅ Human verification for critical alerts

### 4. Testing
- ✅ Test with various lighting conditions
- ✅ Test with different camera angles
- ✅ Test with realistic scenarios
- ✅ Measure false positive rate

## Example Deployment

```bash
# 1. Train extended model
python create_extended_dataset.py
python src/train.py --model_name simple3d --epochs 25

# 2. Test on sample video
python src/infer_video.py \
    --video test_surveillance.mp4 \
    --checkpoint results/checkpoints/best_acc.pth \
    --model_name simple3d \
    --class_names normal suspicious theft fighting weapon_gun weapon_other \
    --threshold 0.4 \
    --output_video output_annotated.mp4

# 3. Deploy real-time
python src/realtime_webcam.py \
    --checkpoint results/checkpoints/best_acc.pth \
    --model_name simple3d \
    --class_names normal suspicious theft fighting weapon_gun weapon_other \
    --threshold 0.4
```

## Legal & Ethical Considerations

⚠️ **Important:**
- Comply with local surveillance laws
- Inform people about monitoring
- Secure storage of footage
- Privacy protection measures
- Human oversight required
- Regular audits for bias

## Support & Resources

- **Datasets:** See EXTENDED_CLASSES.md
- **Training:** See README.md
- **Troubleshooting:** See FAQ.md
- **Windows:** See WINDOWS_SETUP.md

---

**Ready to detect fights and weapons? Start with `python create_extended_dataset.py`**
