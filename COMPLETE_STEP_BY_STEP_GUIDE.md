# Complete Step-by-Step Guide: HAR Theft Detection with 98GB Dataset

## 🎯 Your Situation
- **Dataset**: 98GB (likely UCF-Crime or similar)
- **GCP**: $300 free tier credit
- **Goal**: 90%+ accuracy theft detection system

## ⚠️ IMPORTANT: Don't Download 98GB Locally!

**Why NOT to download locally:**
- ❌ Takes hours/days to download
- ❌ Uses your internet bandwidth
- ❌ Takes up laptop storage
- ❌ Then takes hours to upload to GCP
- ❌ Wastes time and money

**Better approach: Download directly to GCP VM** ✅

---

## 📋 Complete Step-by-Step Process

### Phase 1: GCP Setup (Day 1 - 30 minutes)

#### Step 1.1: Create GCP Project
```bash
# 1. Go to https://console.cloud.google.com/
# 2. Create new project: "har-theft-detection"
# 3. Enable billing (your $300 credit)
# 4. Install gcloud CLI on your laptop
```

**Install gcloud CLI:**
```bash
# Windows (PowerShell)
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe

# Or download from: https://cloud.google.com/sdk/docs/install
```

#### Step 1.2: Setup Project
```bash
# Login and set project
gcloud auth login
gcloud config set project har-theft-detection
gcloud config set compute/zone us-central1-a

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable run.googleapis.com
```

#### Step 1.3: Create Storage Buckets
```bash
# Create buckets for your project
gsutil mb -l us-central1 gs://har-theft-detection-datasets
gsutil mb -l us-central1 gs://har-theft-detection-models
gsutil mb -l us-central1 gs://har-theft-detection-results
```

### Phase 2: VM Setup for Large Dataset (Day 1 - 15 minutes)

#### Step 2.1: Create Powerful VM
```bash
# Create VM with large disk for 98GB dataset
gcloud compute instances create har-data-vm \
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

**Cost**: ~$0.60/hour (with preemptible discount)

#### Step 2.2: SSH into VM
```bash
# Connect to your VM
gcloud compute ssh har-data-vm --zone=us-central1-a
```

### Phase 3: Download Dataset Directly to GCP (Day 1-2)

#### Step 3.1: Setup VM Environment
```bash
# On the VM, update and install tools
sudo apt update
sudo apt install -y wget curl unzip p7zip-full

# Install Python packages
pip install gdown kaggle

# Clone your project
git clone https://github.com/YOUR_USERNAME/har_theft_detection.git
cd har_theft_detection
pip install -r requirements.txt
```

#### Step 3.2: Download Dataset (Choose Your Method)

**Option A: UCF-Crime Dataset**
```bash
# If you have the direct download link
wget "DATASET_DOWNLOAD_URL" -O ucf_crime.zip

# Or if it's on Google Drive
gdown "GOOGLE_DRIVE_FILE_ID" -O ucf_crime.zip
```

**Option B: Kaggle Dataset**
```bash
# Setup Kaggle API (if dataset is on Kaggle)
mkdir ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d DATASET_NAME
```

**Option C: Academic Dataset (Most Common)**
```bash
# Many academic datasets require registration
# 1. Register on the dataset website
# 2. Get download link/credentials
# 3. Use wget or curl with authentication

# Example for UCF-Crime:
wget --user=YOUR_USERNAME --password=YOUR_PASSWORD "DATASET_URL"
```

#### Step 3.3: Extract and Organize Dataset
```bash
# Extract the dataset
unzip ucf_crime.zip -d raw_dataset/
# or
7z x dataset.7z -o raw_dataset/

# Check what you got
ls -la raw_dataset/
du -sh raw_dataset/  # Check size

# Organize by class (this depends on your dataset structure)
mkdir -p data/raw_videos/{Normal,Robbery,Stealing,Burglary,Shoplifting}

# Move videos to appropriate folders
# (This step depends on your specific dataset structure)
```

### Phase 4: Data Preprocessing (Day 2-3)

#### Step 4.1: Extract Clips from Videos
```bash
# This is the most important step - convert videos to clips
python scripts/extract_clips.py \
    --input_dir raw_dataset/ \
    --output_dir data/clips \
    --clip_len 32 \
    --stride 16 \
    --fps 10 \
    --format npy

# This will take several hours for 98GB dataset
# Expected output: 10,000-50,000 clips
```

**Monitor progress:**
```bash
# Check progress in another terminal
watch -n 30 'ls data/clips/ | wc -l'  # Count clips
watch -n 30 'du -sh data/clips/'      # Check size
```

#### Step 4.2: Verify Data Quality
```bash
# Check the generated metadata
head -20 data/metadata_train.csv
wc -l data/metadata_train.csv  # Count training clips
wc -l data/metadata_val.csv    # Count validation clips

# Check class distribution
python -c "
import pandas as pd
df = pd.read_csv('data/metadata_train.csv')
print('Class distribution:')
print(df['label'].value_counts())
print(f'Total clips: {len(df)}')
"
```

#### Step 4.3: Upload Processed Data to Cloud Storage
```bash
# Upload clips to Cloud Storage (for backup and sharing)
gsutil -m cp -r data/clips gs://har-theft-detection-datasets/
gsutil cp data/metadata_*.csv gs://har-theft-detection-datasets/

# This ensures your processed data is safe
```

### Phase 5: Model Training (Day 3-7)

#### Step 5.1: Start with Baseline Model
```bash
# Train Simple3D first (quick test - 1 hour)
python src/train.py \
    --train_csv data/metadata_train.csv \
    --val_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --model_name simple3d \
    --epochs 20 \
    --batch_size 8 \
    --output_dir results/checkpoints/simple3d

# Expected accuracy: 75-80%
```

#### Step 5.2: Train High-Accuracy Models
```bash
# Train SlowFast (3-4 hours)
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
    --output_dir results/checkpoints/slowfast

# Expected accuracy: 85-89%
```

```bash
# Train TimeSformer (4-6 hours) - Best accuracy
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
    --output_dir results/checkpoints/timesformer

# Expected accuracy: 88-92%
```

#### Step 5.3: Monitor Training
```bash
# In another terminal, monitor training
tensorboard --logdir results/logs --host 0.0.0.0 --port 6006

# Access TensorBoard from your laptop:
# http://VM_EXTERNAL_IP:6006
```

### Phase 6: Model Evaluation (Day 7)

#### Step 6.1: Evaluate All Models
```bash
# Evaluate Simple3D
python src/eval.py \
    --test_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --checkpoint results/checkpoints/simple3d/best_acc.pth \
    --model_name simple3d \
    --output_dir results/eval/simple3d

# Evaluate SlowFast
python src/eval.py \
    --test_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --checkpoint results/checkpoints/slowfast/best_acc.pth \
    --model_name slowfast \
    --output_dir results/eval/slowfast

# Evaluate TimeSformer
python src/eval.py \
    --test_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --checkpoint results/checkpoints/timesformer/best_acc.pth \
    --model_name timesformer \
    --output_dir results/eval/timesformer
```

#### Step 6.2: Create Ensemble (Best Accuracy)
```bash
# Ensemble TimeSformer + SlowFast
python src/eval.py \
    --test_csv data/metadata_val.csv \
    --clips_dir data/clips \
    --checkpoint results/checkpoints/timesformer/best_acc.pth \
    --model_name timesformer \
    --ensemble \
    --checkpoint2 results/checkpoints/slowfast/best_acc.pth \
    --model_name2 slowfast \
    --output_dir results/eval/ensemble

# Expected accuracy: 90-95%
```

#### Step 6.3: Download Results to Your Laptop
```bash
# From your laptop, download all results
gcloud compute scp --recurse har-data-vm:~/har_theft_detection/results . --zone=us-central1-a

# Download best models
gcloud compute scp har-data-vm:~/har_theft_detection/results/checkpoints/timesformer/best_acc.pth . --zone=us-central1-a
```

### Phase 7: API Deployment (Day 8)

#### Step 7.1: Prepare API Code
```bash
# On VM, create API directory
mkdir api
cd api

# Create main.py (API code)
cat > main.py << 'EOF'
from flask import Flask, request, jsonify
import torch
import tempfile
import os
import sys
sys.path.append('..')

from src.models import build_model
from src.transforms import VideoTransform, extract_clips_from_video
from src.utils import load_checkpoint

app = Flask(__name__)

# Load model
MODEL = build_model('timesformer', num_classes=3, pretrained=False)
load_checkpoint(MODEL, '../results/checkpoints/timesformer/best_acc.pth', device='cpu')
MODEL.eval()
TRANSFORM = VideoTransform(mode='test')
CLASS_NAMES = ['normal', 'suspicious', 'theft']

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    video_file = request.files['video']
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        video_file.save(temp_video.name)
        
        # Extract clips and run inference
        clips = extract_clips_from_video(temp_video.name, clip_length=32, stride=16, fps=10)
        
        detections = []
        for i, clip_frames in enumerate(clips):
            video_tensor = TRANSFORM(clip_frames).unsqueeze(0)
            
            with torch.no_grad():
                outputs = MODEL(video_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
            
            start_time = i * 16 / 10
            end_time = start_time + 32 / 10
            
            detections.append({
                'start_time': start_time,
                'end_time': end_time,
                'predicted_class': CLASS_NAMES[pred_class],
                'confidence': confidence * 100  # Convert to percentage
            })
        
        os.unlink(temp_video.name)
        
        # Filter alerts
        alerts = [d for d in detections if d['predicted_class'] != 'normal' and d['confidence'] > 50]
        
        return jsonify({
            'success': True,
            'total_clips': len(detections),
            'alerts': len(alerts),
            'detections': alerts
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
EOF
```

#### Step 7.2: Create Dockerfile
```bash
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080
CMD ["python", "main.py"]
EOF
```

#### Step 7.3: Deploy to Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/har-theft-detection/har-api
gcloud run deploy har-api \
    --image gcr.io/har-theft-detection/har-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300
```

### Phase 8: Testing & Demo (Day 9-10)

#### Step 8.1: Test API
```bash
# Get API URL
export API_URL=$(gcloud run services describe har-api --region=us-central1 --format='value(status.url)')

# Test with sample video
curl -X POST -F "video=@test_video.mp4" $API_URL/predict
```

#### Step 8.2: Create Demo Videos
```bash
# Test inference on sample videos
python src/infer_video.py \
    --video sample_video.mp4 \
    --checkpoint results/checkpoints/timesformer/best_acc.pth \
    --model_name timesformer \
    --output_video demo_output.mp4 \
    --output_csv demo_results.csv
```

---

## 💰 Cost Breakdown for 98GB Dataset

### VM Costs (Preemptible):
- **n1-standard-4 + T4 GPU**: $0.60/hour
- **200GB SSD**: $8/month
- **Data processing**: 8-12 hours = $5-7
- **Training**: 10-15 hours = $6-9
- **Total VM cost**: ~$15-20

### Storage Costs:
- **98GB raw dataset**: $2/month
- **Processed clips (~20GB)**: $0.40/month
- **Models and results (~5GB)**: $0.10/month
- **Total storage**: ~$3/month

### API Deployment:
- **Cloud Run**: $2-5/month
- **Container Registry**: $1/month

### **Total Project Cost: $25-35** (out of $300!)

---

## ⏰ Timeline Summary

| Day | Task | Time | Cost |
|-----|------|------|------|
| 1 | GCP setup, VM creation | 1 hour | $1 |
| 1-2 | Download 98GB dataset | 4-8 hours | $3-5 |
| 2-3 | Extract clips | 6-12 hours | $4-7 |
| 3-4 | Train Simple3D + SlowFast | 4-6 hours | $3-4 |
| 4-6 | Train TimeSformer | 6-8 hours | $4-5 |
| 7 | Evaluation + Ensemble | 2-3 hours | $1-2 |
| 8 | API deployment | 2 hours | $1 |
| 9-10 | Testing + Demo | 2 hours | $1 |
| **Total** | **10 days** | **30-45 hours** | **$18-30** |

---

## 🚨 Important Tips

### 1. **Don't Stop VM During Long Operations**
```bash
# For dataset download and clip extraction, keep VM running
# Use screen or tmux to run in background
screen -S dataset_download
# Run your commands
# Ctrl+A, D to detach
# screen -r dataset_download to reattach
```

### 2. **Monitor Costs**
```bash
# Check spending daily
gcloud billing accounts list
gcloud billing projects describe har-theft-detection
```

### 3. **Backup Important Data**
```bash
# Upload processed data to Cloud Storage
gsutil -m cp -r data/clips gs://har-theft-detection-datasets/
gsutil -m cp -r results gs://har-theft-detection-results/
```

### 4. **Use Preemptible Instances**
```bash
# Always use --preemptible for 60-80% savings
# Save checkpoints frequently in case VM gets preempted
```

---

## 🎯 Expected Final Results

With 98GB of real surveillance data, you should achieve:

- **Simple3D**: 75-80% accuracy
- **SlowFast**: 85-89% accuracy  
- **TimeSformer**: 88-92% accuracy
- **Ensemble**: **90-95% accuracy** ✅

**Your goal of 90%+ accuracy is definitely achievable!**

---

## 🚀 Start Today!

**Step 1: Create GCP project (5 minutes)**
```bash
# Go to https://console.cloud.google.com/
# Create project: har-theft-detection
# Enable billing with your $300 credit
```

**Step 2: Create VM (2 minutes)**
```bash
gcloud compute instances create har-data-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --preemptible
```

**Step 3: SSH and start downloading (1 minute)**
```bash
gcloud compute ssh har-data-vm --zone=us-central1-a
```

**You'll be up and running in under 10 minutes!** 🚀

The key is to download the 98GB dataset **directly to the GCP VM**, not to your laptop. This saves hours of time and bandwidth costs.

Need help with any specific step? Just ask! 💪