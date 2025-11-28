# GCP Free Tier Guide for HAR Theft Detection ($300 Credit)

## 💰 Maximizing Your $300 GCP Free Tier Credit

Perfect! With $300 credit for 3 months, you can build and deploy a complete HAR theft detection system. Here's how to use it efficiently:

---

## 📊 Budget Breakdown ($300 over 3 months)

### Recommended Allocation:
- **Training**: $100 (33%) - Train multiple high-accuracy models
- **API Deployment**: $50 (17%) - Deploy inference API
- **Storage**: $30 (10%) - Store models, videos, results
- **Experimentation**: $70 (23%) - Try different configurations
- **Buffer**: $50 (17%) - Safety margin

---

## 🚀 Phase 1: Training Phase ($100 budget)

### GPU Training on Compute Engine

**Recommended VM Configuration:**
```bash
# Cost-effective GPU setup
gcloud compute instances create har-training-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-2 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB \
    --preemptible \
    --maintenance-policy=TERMINATE
```

**Cost Breakdown:**
- **n1-standard-2 + T4 GPU**: ~$0.95/hour
- **Preemptible discount**: ~60% savings = $0.38/hour
- **50GB boot disk**: ~$2/month
- **Training time**: 10-15 hours total

**Training Schedule (Total: ~$15-20):**

**Week 1: Baseline Models**
```bash
# Simple3D (1 hour) - $0.38
python src/train.py --model_name simple3d --epochs 20 --batch_size 8

# R3D (2 hours) - $0.76
python src/train.py --model_name r3d --epochs 30 --batch_size 6
```

**Week 2: High-Accuracy Models**
```bash
# SlowFast (3 hours) - $1.14
python src/train.py --model_name slowfast --epochs 40 --batch_size 8

# TimeSformer (4 hours) - $1.52
python src/train.py --model_name timesformer --epochs 50 --batch_size 4
```

**Week 3: Optimization**
```bash
# Fine-tune best model (3 hours) - $1.14
python src/train.py --model_name timesformer --epochs 60 --lr 5e-5 --batch_size 4

# Ensemble evaluation (1 hour) - $0.38
python src/eval.py --ensemble
```

**Total Training Cost: ~$6-8** (leaves $92-94 for other uses!)

### Money-Saving Tips:

**1. Use Preemptible Instances**
- 60-80% cheaper than regular instances
- Perfect for training (can handle interruptions)
- Save checkpoints frequently

**2. Stop VM When Not Training**
```bash
# Stop VM (saves compute costs, keeps disk)
gcloud compute instances stop har-training-vm --zone=us-central1-a

# Start when needed
gcloud compute instances start har-training-vm --zone=us-central1-a
```

**3. Use Smaller Disk**
- 50GB is enough for training
- Delete large datasets after use
- Use Cloud Storage for long-term storage

---

## 🌐 Phase 2: API Deployment ($50 budget)

### Deploy to Cloud Run (Serverless)

**Why Cloud Run for Free Tier:**
- Pay only for requests (not idle time)
- 2 million requests/month free
- Auto-scaling (0 to 1000+ instances)
- Perfect for demos and testing

**Deployment:**
```bash
# Build and deploy (one-time cost ~$2)
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/har-api

gcloud run deploy har-api \
    --image gcr.io/YOUR_PROJECT_ID/har-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10
```

**Cost Estimation:**
- **Build**: ~$0.003/minute = $0.30 total
- **Container Registry**: ~$0.10/GB/month = $2/month
- **Cloud Run**: First 2M requests free, then $0.40/million
- **Total monthly**: ~$3-5

**Free Tier Limits:**
- ✅ 2 million requests/month (plenty for testing)
- ✅ 360,000 GB-seconds/month
- ✅ 180,000 CPU-seconds/month

### API Usage Examples:

**Test your API:**
```bash
# Get API URL
export API_URL=$(gcloud run services describe har-api --region=us-central1 --format='value(status.url)')

# Test with video
curl -X POST -F "video=@test_video.mp4" $API_URL/predict
```

**Expected Response:**
```json
{
  "success": true,
  "total_clips": 15,
  "alerts": 2,
  "detections": [
    {
      "start_time": 12.5,
      "end_time": 15.7,
      "predicted_class": "theft",
      "confidence": 87.3
    }
  ]
}
```

---

## 💾 Phase 3: Storage Setup ($30 budget)

### Cloud Storage Buckets

**Create buckets:**
```bash
# Models and checkpoints
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-har-models

# Video uploads
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-video-uploads

# Results and logs
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-results
```

**Storage Costs:**
- **Standard Storage**: $0.020/GB/month
- **Nearline Storage**: $0.010/GB/month (for archives)
- **Network egress**: First 1GB/month free

**Storage Plan:**
- **Models**: ~2GB = $0.04/month
- **Videos**: ~10GB = $0.20/month
- **Results**: ~1GB = $0.02/month
- **Total**: ~$0.30/month

### Firestore Database (Free Tier)

**Free tier includes:**
- ✅ 1GB storage
- ✅ 50,000 reads/day
- ✅ 20,000 writes/day
- ✅ 20,000 deletes/day

**Perfect for storing:**
- Detection results
- Video metadata
- User sessions
- Alert logs

---

## 🔬 Phase 4: Experimentation ($70 budget)

### A/B Testing Different Models

**Experiment 1: Model Comparison ($20)**
```bash
# Train 3 models and compare (6 hours total)
python src/train.py --model_name simple3d --epochs 15
python src/train.py --model_name slowfast --epochs 25  
python src/train.py --model_name timesformer --epochs 35
```

**Experiment 2: Hyperparameter Tuning ($25)**
```bash
# Try different learning rates (8 hours total)
python src/train.py --model_name timesformer --lr 5e-5 --epochs 30
python src/train.py --model_name timesformer --lr 1e-4 --epochs 30
python src/train.py --model_name timesformer --lr 2e-4 --epochs 30
```

**Experiment 3: Data Augmentation ($15)**
```bash
# Test different augmentation strategies (5 hours total)
python src/train.py --model_name timesformer --epochs 25 --batch_size 4
# Modify transforms.py and test variations
```

**Experiment 4: Ensemble Methods ($10)**
```bash
# Test ensemble combinations (3 hours total)
python src/eval.py --ensemble --model1 timesformer --model2 slowfast
python src/eval.py --ensemble --model1 timesformer --model2 r3d
```

---

## 📊 Free Tier Monitoring

### Track Your Usage

**Set up billing alerts:**
```bash
# Create budget alert at $50, $100, $150, $200
gcloud billing budgets create \
    --billing-account=YOUR_BILLING_ACCOUNT \
    --display-name="HAR Project Budget" \
    --budget-amount=50USD \
    --threshold-rule=percent=80 \
    --threshold-rule=percent=100
```

**Monitor costs:**
```bash
# Check current usage
gcloud billing budgets list

# View detailed billing
gcloud logging read "protoPayload.serviceName=compute.googleapis.com" --limit=50
```

### Cost Optimization Tips:

**1. Use Preemptible Instances (60-80% savings)**
```bash
--preemptible  # Add to all compute instances
```

**2. Stop Resources When Not Needed**
```bash
# Stop VM
gcloud compute instances stop har-training-vm --zone=us-central1-a

# Delete when done
gcloud compute instances delete har-training-vm --zone=us-central1-a
```

**3. Use Regional Resources**
```bash
--region=us-central1  # Cheaper than multi-region
```

**4. Clean Up Regularly**
```bash
# Delete old images
gcloud compute images list
gcloud compute images delete OLD_IMAGE_NAME

# Clean up storage
gsutil rm -r gs://bucket-name/old-data/
```

---

## 🎯 3-Month Timeline

### Month 1: Setup & Training ($80)
- **Week 1**: Setup GCP, create VM, train baseline models
- **Week 2**: Train TimeSformer and SlowFast
- **Week 3**: Optimize best model, create ensemble
- **Week 4**: Deploy API to Cloud Run

### Month 2: Optimization & Testing ($100)
- **Week 1**: Hyperparameter tuning
- **Week 2**: Data augmentation experiments  
- **Week 3**: Model architecture comparisons
- **Week 4**: Performance optimization

### Month 3: Production & Demo ($120)
- **Week 1**: Real-time processing setup
- **Week 2**: Monitoring and alerting
- **Week 3**: Frontend dashboard
- **Week 4**: Final testing and documentation

---

## 💡 Free Tier Maximization Strategies

### 1. **Always On Free Tier Services**
These don't count against your $300:
- ✅ Cloud Functions: 2M invocations/month
- ✅ Cloud Run: 2M requests/month
- ✅ Firestore: 1GB storage + 50K reads/day
- ✅ Cloud Storage: 5GB/month
- ✅ BigQuery: 1TB queries/month

### 2. **Stackdriver Monitoring (Free)**
```bash
# Monitor everything for free
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
```

### 3. **Use Cloud Shell (Free)**
```bash
# Free development environment
# 5GB persistent disk
# Pre-installed tools
```

---

## 🚨 Budget Alerts Setup

**Create smart alerts:**
```bash
# Alert at 50% usage
gcloud alpha billing budgets create \
    --billing-account=$BILLING_ACCOUNT_ID \
    --display-name="HAR 50% Alert" \
    --budget-amount=150USD \
    --threshold-rule=percent=100

# Alert at 80% usage  
gcloud alpha billing budgets create \
    --billing-account=$BILLING_ACCOUNT_ID \
    --display-name="HAR 80% Alert" \
    --budget-amount=240USD \
    --threshold-rule=percent=100
```

---

## 📈 Expected Results with $300 Budget

### What You Can Achieve:

**Models Trained:**
- ✅ 5+ different architectures
- ✅ 10+ hyperparameter experiments
- ✅ 1-2 high-accuracy ensemble models
- ✅ Fully optimized TimeSformer (90%+ accuracy)

**Infrastructure Deployed:**
- ✅ Production-ready API (Cloud Run)
- ✅ Real-time processing (Cloud Functions)
- ✅ Storage and database (Cloud Storage + Firestore)
- ✅ Monitoring and alerts

**Final Deliverables:**
- ✅ Trained models with 90%+ accuracy
- ✅ REST API for video analysis
- ✅ Web dashboard for monitoring
- ✅ Complete documentation
- ✅ Demo videos and presentations

---

## 🎓 Pro Tips for Free Tier

### 1. **Start Small, Scale Up**
```bash
# Begin with small instances
--machine-type=n1-standard-1  # $0.05/hour

# Upgrade only when needed
--machine-type=n1-standard-4  # $0.19/hour
```

### 2. **Use Spot/Preemptible Instances**
```bash
--preemptible  # 60-80% discount
```

### 3. **Automate Shutdown**
```bash
# Auto-shutdown after training
gcloud compute instances add-metadata har-training-vm \
    --metadata=shutdown-script="sudo shutdown -h now"
```

### 4. **Monitor Costs Daily**
```bash
# Check spending
gcloud billing accounts list
gcloud billing projects describe $PROJECT_ID
```

---

## 🚀 Quick Start for Free Tier

**Day 1: Setup (Free)**
```bash
# Create project and enable APIs
gcloud projects create har-theft-detection
gcloud config set project har-theft-detection
gcloud services enable compute.googleapis.com
gcloud services enable run.googleapis.com
```

**Day 2: First Training ($2)**
```bash
# Create preemptible VM and train Simple3D
gcloud compute instances create har-vm --preemptible --accelerator=type=nvidia-tesla-t4,count=1
# SSH and train for 2 hours
```

**Day 3: Deploy API ($1)**
```bash
# Deploy to Cloud Run
gcloud run deploy har-api --allow-unauthenticated
```

**Total spent: $3 out of $300** ✅

---

## 📞 Need Help?

### Free Resources:
- **GCP Documentation**: Free and comprehensive
- **Qwiklabs**: Free credits for hands-on labs
- **Google Cloud Community**: Free support forum
- **YouTube**: Google Cloud Tech channel

### Cost Monitoring:
- Set up billing alerts immediately
- Check usage daily
- Use preemptible instances
- Clean up resources regularly

---

**Your $300 free tier credit is more than enough to build a complete, production-ready HAR theft detection system! 🎉**

**Estimated final cost: $150-200** (leaving you $100-150 buffer)

Start with the quick setup and you'll have a working system within days!