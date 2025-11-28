# Dropbox Dataset Setup Guide for HAR Suspicious Activities

## 🎯 Your Dataset Overview

**Dropbox Link**: https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&st=3tu2jxkj&dl=0

**Advantages of Your Dataset:**
- ✅ **Immediate access** - No registration required
- ✅ **Direct download** - Can download straight to GCP VM
- ✅ **Likely pre-organized** - Probably already structured for HAR
- ✅ **Custom for suspicious activities** - Perfect for your project title

---

## 🚀 Step-by-Step Setup with Your Dropbox Dataset

### Step 1: Create GCP VM (5 minutes)

```bash
# Set up GCP project
gcloud config set project har-theft-detection
gcloud config set compute/zone us-central1-a

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com

# Create VM with large storage for your dataset
gcloud compute instances create har-suspicious-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --preemptible \
    --maintenance-policy=TERMINATE
```

**Cost**: ~$0.60/hour with preemptible discount

### Step 2: Setup VM Environment (10 minutes)

```bash
# SSH into VM
gcloud compute ssh har-suspicious-vm --zone=us-central1-a

# Update system and install tools
sudo apt update
sudo apt install -y wget curl unzip p7zip-full tree

# Clone your project
git clone https://github.com/YOUR_USERNAME/har_theft_detection.git
cd har_theft_detection

# Install Python dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Download Dataset from Dropbox (1-3 hours)

**Method 1: Direct wget (Recommended)**
```bash
# Create dataset directory
mkdir -p dataset_raw
cd dataset_raw

# Convert Dropbox share link to direct download
# Your link: https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&st=3tu2jxkj&dl=0
# For direct download, change dl=0 to dl=1

DROPBOX_URL="https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&st=3tu2jxkj&dl=1"

# Download the dataset
wget "$DROPBOX_URL" -O suspicious_activities_dataset.zip

# Check download
ls -lh suspicious_activities_dataset.zip
```

**Method 2: Using Dropbox CLI (Alternative)**
```bash
# Install Dropbox CLI
wget -O - "https://www.dropbox.com/download?plat=lnx.x86_64" | tar xzf -
sudo mv .dropbox-dist/dropboxd /usr/local/bin/
dropbox start -i

# Download using Dropbox CLI (if you have access)
# This method requires Dropbox account authentication
```

**Method 3: Manual Download Script**
```bash
# Create download script for large files
cat > download_dataset.py << 'EOF'
import requests
import os
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

# Your Dropbox URL (with dl=1 for direct download)
url = "https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&st=3tu2jxkj&dl=1"
filename = "suspicious_activities_dataset.zip"

print(f"Downloading {filename}...")
download_file(url, filename)
print("Download completed!")
EOF

# Run the download script
python download_dataset.py
```

### Step 4: Extract and Explore Dataset (30 minutes)

```bash
# Extract the dataset
unzip suspicious_activities_dataset.zip -d extracted_dataset/

# Explore the structure
cd extracted_dataset/
tree -L 3 .

# Check what we have
echo "Dataset structure:"
find . -type d -name "*" | head -20

echo "Video files found:"
find . -name "*.mp4" -o -name "*.avi" -o -name "*.mov" | wc -l

echo "File sizes:"
du -sh */
```

### Step 5: Organize Dataset for HAR System (1 hour)

**Based on common dataset structures, your dataset likely has one of these formats:**

**Format A: Already organized by class**
```
dataset/
├── normal/
│   ├── video1.mp4
│   └── video2.mp4
├── suspicious/
│   ├── video1.mp4
│   └── video2.mp4
└── theft/
    ├── video1.mp4
    └── video2.mp4
```

**Format B: Single folder with labeled files**
```
dataset/
├── normal_001.mp4
├── normal_002.mp4
├── suspicious_001.mp4
├── suspicious_002.mp4
├── theft_001.mp4
└── theft_002.mp4
```

**Format C: Annotation files**
```
dataset/
├── videos/
│   ├── video1.mp4
│   └── video2.mp4
└── annotations/
    ├── labels.csv
    └── metadata.json
```

**Organize based on your dataset structure:**

```bash
# Go back to project root
cd ~/har_theft_detection

# Create organized structure
mkdir -p data/raw_videos/{normal,suspicious,theft}

# Method 1: If already organized by folders
if [ -d "extracted_dataset/normal" ]; then
    echo "Dataset already organized by class"
    cp extracted_dataset/normal/*.* data/raw_videos/normal/
    cp extracted_dataset/suspicious/*.* data/raw_videos/suspicious/
    cp extracted_dataset/theft/*.* data/raw_videos/theft/
fi

# Method 2: If files are named by class
if [ ! -d "extracted_dataset/normal" ]; then
    echo "Organizing files by name patterns"
    find extracted_dataset/ -name "*normal*" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) -exec cp {} data/raw_videos/normal/ \;
    find extracted_dataset/ -name "*suspicious*" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) -exec cp {} data/raw_videos/suspicious/ \;
    find extracted_dataset/ -name "*theft*" -o -name "*steal*" -o -name "*rob*" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) -exec cp {} data/raw_videos/theft/ \;
fi

# Check organized structure
echo "Organized dataset:"
for class in normal suspicious theft; do
    count=$(ls data/raw_videos/$class/*.* 2>/dev/null | wc -l)
    echo "$class: $count videos"
done
```

### Step 6: Extract Clips from Your Dataset (2-4 hours)

```bash
# Extract clips using our preprocessing script
python scripts/extract_clips.py \
    --input_dir data/raw_videos \
    --output_dir data/clips \
    --clip_len 32 \
    --stride 16 \
    --fps 10 \
    --format npy \
    --min_clips 1

# Monitor progress
watch -n 60 'echo "Clips: $(ls data/clips/*.npy 2>/dev/null | wc -l), Size: $(du -sh data/clips/ 2>/dev/null)"'
```

**Expected output:**
```
Extraction completed!
Total clips: 5,000-15,000 (depends on your dataset size)
Train clips: 4,000-12,000 -> data/metadata_train.csv
Val clips: 1,000-3,000 -> data/metadata_val.csv

Class distribution:
normal        2000-6000
suspicious    1500-4000
theft         1500-5000
```

### Step 7: Verify Dataset Quality

```bash
# Check metadata
head -10 data/metadata_train.csv
echo "Training clips: $(wc -l < data/metadata_train.csv)"
echo "Validation clips: $(wc -l < data/metadata_val.csv)"

# Check class distribution
python -c "
import pandas as pd
df = pd.read_csv('data/metadata_train.csv')
print('Training set class distribution:')
print(df['label'].value_counts())
print(f'Total training clips: {len(df)}')

df_val = pd.read_csv('data/metadata_val.csv')
print('\nValidation set class distribution:')  
print(df_val['label'].value_counts())
print(f'Total validation clips: {len(df_val)}')
"

# Test loading clips
python -c "
from src.datasets import VideoClipDataset
from src.transforms import VideoTransform

try:
    dataset = VideoClipDataset(
        metadata_csv='data/metadata_train.csv',
        clips_dir='data/clips',
        num_frames=32,
        mode='train'
    )
    
    print(f'✅ Dataset loaded successfully!')
    print(f'Classes: {dataset.labels}')
    print(f'Total samples: {len(dataset)}')
    
    # Test a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f'Sample {i}: video {sample[\"video\"].shape}, label {sample[\"label\"]}')
        
except Exception as e:
    print(f'❌ Error loading dataset: {e}')
"
```

---

## 🚀 Training on Your Dropbox Dataset

### Phase 1: Quick Baseline (1 hour, $0.60)

```bash
# Train Simple3D for quick validation
python src/train.py \
    --train_csv data/metadata_train.csv \
    --val_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --model_name simple3d \
    --epochs 15 \
    --batch_size 8 \
    --lr 3e-4 \
    --output_dir results/checkpoints/simple3d_dropbox

# Expected: 70-80% accuracy (good baseline)
```

### Phase 2: High-Accuracy Models (6-10 hours, $4-6)

```bash
# Train SlowFast
python src/train.py \
    --train_csv data/metadata_train.csv \
    --val_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --model_name slowfast \
    --epochs 40 \
    --batch_size 8 \
    --lr 3e-4 \
    --use_class_weights \
    --use_amp \
    --output_dir results/checkpoints/slowfast_dropbox

# Expected: 82-88% accuracy
```

```bash
# Train TimeSformer (best single model)
python src/train.py \
    --train_csv data/metadata_train.csv \
    --val_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --model_name timesformer \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --use_class_weights \
    --use_amp \
    --output_dir results/checkpoints/timesformer_dropbox

# Expected: 85-92% accuracy
```

### Phase 3: Ensemble for Maximum Accuracy (1 hour, $0.60)

```bash
# Create ensemble for best results
python src/eval.py \
    --test_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --checkpoint results/checkpoints/timesformer_dropbox/best_acc.pth \
    --model_name timesformer \
    --ensemble \
    --checkpoint2 results/checkpoints/slowfast_dropbox/best_acc.pth \
    --model_name2 slowfast \
    --output_dir results/eval/ensemble_dropbox

# Expected: 88-95% accuracy ✅
```

---

## 📊 Expected Results with Your Dataset

### Performance Estimates (depends on dataset quality):

| Model | Expected Accuracy | Training Time | Cost |
|-------|------------------|---------------|------|
| Simple3D | 70-80% | 1 hour | $0.60 |
| SlowFast | 82-88% | 4 hours | $2.40 |
| TimeSformer | 85-92% | 6 hours | $3.60 |
| **Ensemble** | **88-95%** | - | - |

### Factors Affecting Accuracy:
- **Dataset size**: More videos = better accuracy
- **Video quality**: HD videos = better features
- **Label quality**: Accurate labels = better training
- **Class balance**: Balanced classes = better performance

---

## 💰 Cost Breakdown

### VM Costs (Preemptible n1-standard-4 + T4):
- **Dataset download**: 2-4 hours × $0.60 = $1.20-2.40
- **Data processing**: 4-6 hours × $0.60 = $2.40-3.60
- **Training**: 10-12 hours × $0.60 = $6.00-7.20
- **Total VM**: $10-15

### Storage Costs:
- **200GB SSD**: $8/month
- **Cloud Storage backup**: $1-2/month

### **Total Project Cost: $15-25** (out of $300!)

---

## 🔧 Troubleshooting Your Dataset

### Issue 1: Download Fails
```bash
# Try alternative download methods
curl -L "$DROPBOX_URL" -o dataset.zip
# or
python download_dataset.py
```

### Issue 2: Unknown Dataset Structure
```bash
# Explore thoroughly
find extracted_dataset/ -type f -name "*.txt" -o -name "*.csv" -o -name "*.json" | head -10
cat extracted_dataset/README.txt  # if exists
```

### Issue 3: Mixed File Formats
```bash
# Convert all videos to MP4
for file in data/raw_videos/*/*.{avi,mov,mkv}; do
    if [ -f "$file" ]; then
        ffmpeg -i "$file" "${file%.*}.mp4"
        rm "$file"
    fi
done
```

### Issue 4: Large Video Files
```bash
# Compress videos if too large
for file in data/raw_videos/*/*.mp4; do
    if [ $(stat -c%s "$file") -gt 100000000 ]; then  # >100MB
        ffmpeg -i "$file" -vcodec libx264 -crf 28 "${file%.*}_compressed.mp4"
        mv "${file%.*}_compressed.mp4" "$file"
    fi
done
```

---

## 📋 Quick Start Commands for Your Dataset

```bash
# 1. Create VM
gcloud compute instances create har-suspicious-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=200GB \
    --preemptible \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release

# 2. SSH and setup
gcloud compute ssh har-suspicious-vm --zone=us-central1-a
git clone https://github.com/YOUR_USERNAME/har_theft_detection.git
cd har_theft_detection
pip install -r requirements.txt

# 3. Download your dataset
mkdir dataset_raw && cd dataset_raw
wget "https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&st=3tu2jxkj&dl=1" -O dataset.zip
unzip dataset.zip

# 4. Organize and process
cd ~/har_theft_detection
# Organize videos into data/raw_videos/{normal,suspicious,theft}/
python scripts/extract_clips.py --input_dir data/raw_videos --output_dir data/clips

# 5. Train models
python src/train.py --model_name timesformer --epochs 50 --batch_size 4
```

---

## 🎯 Next Steps

1. **Start VM creation** (5 minutes)
2. **Download your dataset** (1-3 hours)
3. **Explore and organize** (1 hour)
4. **Extract clips** (2-4 hours)
5. **Train models** (6-10 hours)
6. **Achieve 90%+ accuracy** with ensemble! ✅

**Your Dropbox dataset is perfect because:**
- ✅ **No registration required** - Start immediately
- ✅ **Direct download** - Faster than UCF-Crime
- ✅ **Likely optimized** - Probably already prepared for HAR
- ✅ **Custom for suspicious activities** - Perfect for your project

**Ready to start? The dataset download is the first step!** 🚀