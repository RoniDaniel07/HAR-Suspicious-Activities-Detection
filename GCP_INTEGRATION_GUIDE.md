# Google Cloud Platform (GCP) Integration Guide

## 🌐 Complete GCP Integration for HAR Theft Detection System

This guide shows you how to deploy, train, and run your theft detection system on Google Cloud Platform for scalable, production-ready deployment.

---

## 🎯 GCP Integration Options

### 1. **Training on GCP** (Compute Engine with GPU)
- Train models on powerful cloud GPUs
- Scale up/down as needed
- Pay only for what you use

### 2. **Inference API** (Cloud Run + Cloud Functions)
- Deploy model as REST API
- Auto-scaling serverless inference
- Handle video uploads and processing

### 3. **Real-time Processing** (Pub/Sub + Cloud Functions)
- Process live video streams
- Real-time alerts and notifications
- Integration with security systems

### 4. **Storage & Database** (Cloud Storage + Firestore)
- Store videos, models, and results
- Scalable database for detections
- Backup and versioning

### 5. **Monitoring & Alerts** (Cloud Monitoring + Alerting)
- Monitor model performance
- Send alerts when theft detected
- Dashboard for security teams

---

## 🚀 Option 1: Training on GCP Compute Engine

### Setup GCP VM with GPU

**Create VM with GPU:**
```bash
# Set project and zone
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/zone us-central1-a

# Create VM with GPU
gcloud compute instances create har-training-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

**SSH into VM:**
```bash
gcloud compute ssh har-training-vm --zone=us-central1-a
```

**Setup on VM:**
```bash
# Clone your project
git clone https://github.com/YOUR_USERNAME/har_theft_detection.git
cd har_theft_detection

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Train model
python src/train.py --model_name timesformer --epochs 50 --batch_size 4
```

**Download trained model:**
```bash
# From your local machine
gcloud compute scp har-training-vm:~/har_theft_detection/results/checkpoints/best_acc.pth . --zone=us-central1-a
```

---

## 🌐 Option 2: Deploy as REST API (Cloud Run)

### Create Inference API

**Create `api/main.py`:**
```python
from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import tempfile
import os
from google.cloud import storage
import sys
sys.path.append('.')

from src.models import build_model
from src.transforms import VideoTransform, extract_clips_from_video
from src.utils import load_checkpoint

app = Flask(__name__)

# Load model once at startup
MODEL = None
TRANSFORM = None
CLASS_NAMES = ['normal', 'suspicious', 'theft']

def load_model():
    global MODEL, TRANSFORM
    
    # Download model from Cloud Storage
    client = storage.Client()
    bucket = client.bucket('your-har-models-bucket')
    blob = bucket.blob('models/best_timesformer.pth')
    
    with tempfile.NamedTemporaryFile() as temp_file:
        blob.download_to_filename(temp_file.name)
        
        MODEL = build_model('timesformer', num_classes=3, pretrained=False)
        load_checkpoint(MODEL, temp_file.name, device='cpu')
        MODEL.eval()
    
    TRANSFORM = VideoTransform(mode='test')
    print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': MODEL is not None})

@app.route('/predict', methods=['POST'])
def predict_video():
    try:
        # Get uploaded video
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            video_file.save(temp_video.name)
            
            # Extract clips
            clips = extract_clips_from_video(
                temp_video.name,
                clip_length=32,
                stride=16,
                fps=10
            )
            
            # Run inference
            detections = []
            
            for i, clip_frames in enumerate(clips):
                video_tensor = TRANSFORM(clip_frames)
                video_tensor = video_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    outputs = MODEL(video_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    pred_class = torch.argmax(probs).item()
                    confidence = probs[pred_class].item()
                
                # Calculate timestamps
                start_time = i * 16 / 10  # stride / fps
                end_time = start_time + 32 / 10  # clip_length / fps
                
                detections.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'predicted_class': CLASS_NAMES[pred_class],
                    'confidence': confidence,
                    'probabilities': {
                        CLASS_NAMES[j]: float(probs[j]) 
                        for j in range(len(CLASS_NAMES))
                    }
                })
            
            # Clean up
            os.unlink(temp_video.name)
            
            # Filter suspicious/theft detections
            alerts = [d for d in detections 
                     if d['predicted_class'] != 'normal' and d['confidence'] > 0.5]
            
            return jsonify({
                'success': True,
                'total_clips': len(detections),
                'alerts': len(alerts),
                'detections': alerts,
                'summary': {
                    'normal': sum(1 for d in detections if d['predicted_class'] == 'normal'),
                    'suspicious': sum(1 for d in detections if d['predicted_class'] == 'suspicious'),
                    'theft': sum(1 for d in detections if d['predicted_class'] == 'theft')
                }
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
```

**Create `api/requirements.txt`:**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.0
flask>=2.3.0
google-cloud-storage>=2.10.0
numpy>=1.24.0
```

**Create `api/Dockerfile`:**
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "api/main.py"]
```

**Deploy to Cloud Run:**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/har-api
gcloud run deploy har-api \
    --image gcr.io/YOUR_PROJECT_ID/har-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300
```

**Test API:**
```bash
# Get the URL from deployment
export API_URL="https://har-api-xxx-uc.a.run.app"

# Test health
curl $API_URL/health

# Test prediction
curl -X POST -F "video=@test_video.mp4" $API_URL/predict
```

---

## 📡 Option 3: Real-time Processing (Pub/Sub + Functions)

### Setup Pub/Sub for Video Streams

**Create topics:**
```bash
# Create topics
gcloud pubsub topics create video-upload
gcloud pubsub topics create theft-alerts

# Create subscriptions
gcloud pubsub subscriptions create video-processing --topic=video-upload
gcloud pubsub subscriptions create alert-notifications --topic=theft-alerts
```

**Create Cloud Function for processing:**

**`functions/main.py`:**
```python
import functions_framework
from google.cloud import storage, pubsub_v1, firestore
import tempfile
import json
import torch
from datetime import datetime

# Initialize clients
storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()
db = firestore.Client()

@functions_framework.cloud_event
def process_video(cloud_event):
    """Process uploaded video and detect theft"""
    
    # Get file info from event
    data = cloud_event.data
    bucket_name = data['bucket']
    file_name = data['name']
    
    if not file_name.endswith(('.mp4', '.avi', '.mov')):
        return
    
    print(f"Processing video: {file_name}")
    
    try:
        # Download video
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            blob.download_to_filename(temp_file.name)
            
            # Run inference (simplified)
            detections = run_inference(temp_file.name)
            
            # Save results to Firestore
            doc_ref = db.collection('detections').document()
            doc_ref.set({
                'video_file': file_name,
                'timestamp': datetime.utcnow(),
                'detections': detections,
                'alert_count': len([d for d in detections if d['predicted_class'] != 'normal'])
            })
            
            # Send alerts if theft detected
            alerts = [d for d in detections if d['predicted_class'] == 'theft']
            if alerts:
                send_alert(file_name, alerts)
                
    except Exception as e:
        print(f"Error processing video: {e}")

def run_inference(video_path):
    """Run theft detection inference"""
    # Load model and run inference
    # (Implementation similar to API version)
    pass

def send_alert(video_file, alerts):
    """Send theft alert via Pub/Sub"""
    topic_path = publisher.topic_path('YOUR_PROJECT_ID', 'theft-alerts')
    
    alert_data = {
        'video_file': video_file,
        'timestamp': datetime.utcnow().isoformat(),
        'alert_type': 'THEFT_DETECTED',
        'detections': alerts
    }
    
    publisher.publish(topic_path, json.dumps(alert_data).encode('utf-8'))
```

**Deploy function:**
```bash
gcloud functions deploy process-video \
    --runtime python39 \
    --trigger-bucket your-video-uploads-bucket \
    --memory 2GB \
    --timeout 300s
```

---

## 💾 Option 4: Storage & Database Setup

### Cloud Storage Setup

```bash
# Create buckets
gsutil mb gs://your-har-models-bucket
gsutil mb gs://your-video-uploads-bucket
gsutil mb gs://your-processed-videos-bucket

# Upload trained model
gsutil cp results/checkpoints/best_acc.pth gs://your-har-models-bucket/models/

# Set permissions
gsutil iam ch allUsers:objectViewer gs://your-har-models-bucket
```

### Firestore Database Structure

```javascript
// Collection: detections
{
  "video_file": "security_cam_001_20241127.mp4",
  "timestamp": "2024-11-27T10:30:00Z",
  "camera_id": "CAM_001",
  "location": "Store Entrance",
  "detections": [
    {
      "start_time": 45.2,
      "end_time": 48.4,
      "predicted_class": "theft",
      "confidence": 0.87,
      "bounding_box": [100, 150, 200, 300]
    }
  ],
  "alert_sent": true,
  "reviewed": false
}

// Collection: cameras
{
  "camera_id": "CAM_001",
  "location": "Store Entrance",
  "status": "active",
  "last_seen": "2024-11-27T10:35:00Z",
  "model_version": "timesformer_v1.0"
}
```

---

## 📊 Option 5: Monitoring & Alerting

### Cloud Monitoring Setup

**Create custom metrics:**
```python
from google.cloud import monitoring_v3

def record_detection_metric(project_id, detection_type, confidence):
    """Record detection metrics"""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    # Create time series data
    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/har/detections"
    series.metric.labels["detection_type"] = detection_type
    series.resource.type = "global"
    
    # Add data point
    point = series.points.add()
    point.value.double_value = confidence
    point.interval.end_time.seconds = int(time.time())
    
    client.create_time_series(name=project_name, time_series=[series])
```

**Setup alerting policy:**
```bash
# Create alerting policy for theft detection
gcloud alpha monitoring policies create \
    --policy-from-file=alerting-policy.yaml
```

**`alerting-policy.yaml`:**
```yaml
displayName: "Theft Detection Alert"
conditions:
  - displayName: "High confidence theft detected"
    conditionThreshold:
      filter: 'metric.type="custom.googleapis.com/har/detections" AND metric.label.detection_type="theft"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 0.8
      duration: 60s
notificationChannels:
  - "projects/YOUR_PROJECT_ID/notificationChannels/YOUR_CHANNEL_ID"
```

---

## 🔧 Complete Deployment Script

**Create `deploy_gcp.sh`:**
```bash
#!/bin/bash

# Set variables
PROJECT_ID="your-project-id"
REGION="us-central1"
ZONE="us-central1-a"

echo "Deploying HAR Theft Detection System to GCP..."

# 1. Setup project
gcloud config set project $PROJECT_ID
gcloud services enable compute.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable pubsub.googleapis.com

# 2. Create storage buckets
gsutil mb gs://$PROJECT_ID-har-models
gsutil mb gs://$PROJECT_ID-video-uploads
gsutil mb gs://$PROJECT_ID-processed-videos

# 3. Upload model (if you have one trained)
if [ -f "results/checkpoints/best_acc.pth" ]; then
    gsutil cp results/checkpoints/best_acc.pth gs://$PROJECT_ID-har-models/models/
fi

# 4. Deploy API to Cloud Run
gcloud builds submit --tag gcr.io/$PROJECT_ID/har-api
gcloud run deploy har-api \
    --image gcr.io/$PROJECT_ID/har-api \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300

# 5. Create Pub/Sub topics
gcloud pubsub topics create video-upload
gcloud pubsub topics create theft-alerts

# 6. Deploy Cloud Function
gcloud functions deploy process-video \
    --runtime python39 \
    --trigger-bucket $PROJECT_ID-video-uploads \
    --memory 2GB \
    --timeout 300s

echo "Deployment complete!"
echo "API URL: $(gcloud run services describe har-api --region=$REGION --format='value(status.url)')"
```

---

## 📱 Frontend Integration

### Web Dashboard

**Create `dashboard/index.html`:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>HAR Theft Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Security Monitoring Dashboard</h1>
    
    <div id="upload-section">
        <h2>Upload Video for Analysis</h2>
        <input type="file" id="videoFile" accept="video/*">
        <button onclick="uploadVideo()">Analyze Video</button>
    </div>
    
    <div id="results"></div>
    
    <script>
        const API_URL = 'https://your-har-api-url.run.app';
        
        async function uploadVideo() {
            const fileInput = document.getElementById('videoFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a video file');
                return;
            }
            
            const formData = new FormData();
            formData.append('video', file);
            
            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                console.error('Error:', error);
                alert('Error analyzing video');
            }
        }
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            
            if (result.alerts && result.alerts.length > 0) {
                resultsDiv.innerHTML = `
                    <h2>⚠️ ALERTS DETECTED</h2>
                    <p>Found ${result.alerts.length} suspicious activities</p>
                    <ul>
                        ${result.alerts.map(alert => `
                            <li>
                                ${alert.predicted_class.toUpperCase()} 
                                at ${alert.start_time.toFixed(1)}s - ${alert.end_time.toFixed(1)}s
                                (Confidence: ${(alert.confidence * 100).toFixed(1)}%)
                            </li>
                        `).join('')}
                    </ul>
                `;
            } else {
                resultsDiv.innerHTML = '<h2>✅ No threats detected</h2>';
            }
        }
    </script>
</body>
</html>
```

---

## 💰 Cost Estimation

### Training Costs (one-time)
- **Compute Engine (n1-standard-4 + T4 GPU)**: ~$1.50/hour
- **Training TimeSformer (4 hours)**: ~$6
- **Storage (100GB)**: ~$2/month

### Inference Costs (ongoing)
- **Cloud Run**: $0.000024/request + $0.000009/GB-second
- **Cloud Functions**: $0.0000004/invocation + $0.0000025/GB-second
- **Cloud Storage**: $0.020/GB/month
- **Firestore**: $0.18/100K reads, $0.18/100K writes

### Example Monthly Costs (1000 videos/month)
- **API calls**: ~$5
- **Storage**: ~$10
- **Database**: ~$5
- **Total**: ~$20/month

---

## 🚀 Quick Start Commands

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/har_theft_detection.git
cd har_theft_detection

# 2. Set GCP project
gcloud config set project YOUR_PROJECT_ID

# 3. Deploy everything
chmod +x deploy_gcp.sh
./deploy_gcp.sh

# 4. Test API
curl -X POST -F "video=@test_video.mp4" https://your-api-url.run.app/predict
```

---

## 📚 Additional Resources

### GCP Documentation
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [Compute Engine Documentation](https://cloud.google.com/compute/docs)

### Monitoring & Alerting
- [Cloud Monitoring](https://cloud.google.com/monitoring/docs)
- [Error Reporting](https://cloud.google.com/error-reporting/docs)
- [Cloud Logging](https://cloud.google.com/logging/docs)

---

**Your HAR system is now ready for enterprise-scale deployment on GCP! 🌐**

Choose the integration option that best fits your needs:
- **Option 1**: For training large models
- **Option 2**: For API-based inference
- **Option 3**: For real-time processing
- **Option 4**: For data storage and management
- **Option 5**: For monitoring and alerts

All options can be combined for a complete solution!