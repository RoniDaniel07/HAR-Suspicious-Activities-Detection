#!/bin/bash

# Script to download and prepare datasets for HAR theft detection

echo "=========================================="
echo "HAR Theft Detection - Data Download Script"
echo "=========================================="

# Create directories
mkdir -p data/raw_videos
mkdir -p data/clips
mkdir -p data/datasets

echo ""
echo "This script helps you download public datasets for training."
echo "Available datasets:"
echo "  1. UCF-Crime (Anomaly Detection)"
echo "  2. CCTV-Fights"
echo "  3. Create dummy dataset for testing"
echo ""

read -p "Select option (1-3): " option

case $option in
    1)
        echo ""
        echo "UCF-Crime Dataset"
        echo "-----------------"
        echo "This is a large dataset (~38 GB)"
        echo "Download from: https://www.crcv.ucf.edu/projects/real-world/"
        echo ""
        echo "Manual steps:"
        echo "1. Visit the website and request access"
        echo "2. Download the dataset"
        echo "3. Extract to data/raw_videos/"
        echo "4. Organize by class: data/raw_videos/{normal,robbery,stealing}/"
        echo ""
        echo "Then run:"
        echo "  python scripts/extract_clips.py --input_dir data/raw_videos --output_dir data/clips"
        ;;
    
    2)
        echo ""
        echo "CCTV-Fights Dataset"
        echo "-------------------"
        echo "Download from: https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch"
        echo ""
        echo "Manual steps:"
        echo "1. Clone the repository"
        echo "2. Download videos"
        echo "3. Extract to data/raw_videos/"
        echo ""
        ;;
    
    3)
        echo ""
        echo "Creating dummy dataset for testing..."
        python -c "from src.datasets import create_dummy_dataset; create_dummy_dataset(num_videos_per_class=10)"
        echo ""
        echo "Dummy dataset created!"
        echo "  Train: data/metadata_train.csv"
        echo "  Val: data/metadata_val.csv"
        echo "  Clips: data/clips/"
        echo ""
        echo "You can now train a model:"
        echo "  python src/train.py --model_name simple3d --epochs 10"
        ;;
    
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
