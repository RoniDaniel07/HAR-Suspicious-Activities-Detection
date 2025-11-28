"""
Create extended dataset with 6 classes (including fighting and weapons)
"""
import sys
sys.path.insert(0, '.')

from src.datasets import create_dummy_dataset

print("="*60)
print("Creating Extended Dataset (6 Classes)")
print("="*60)
print()
print("Classes:")
print("  1. Normal - Regular activities")
print("  2. Suspicious - Concerning behavior")
print("  3. Theft - Stealing, robbery")
print("  4. Fighting - Physical altercations")
print("  5. Weapon (Gun) - Firearms")
print("  6. Weapon (Other) - Knives, bombs, etc.")
print()

# Create 6-class dataset
classes = ['normal', 'suspicious', 'theft', 'fighting', 'weapon_gun', 'weapon_other']

train_csv, val_csv = create_dummy_dataset(
    output_dir='data',
    num_videos_per_class=8,  # 8 videos per class = 48 total
    num_frames=32,
    classes=classes
)

print()
print("="*60)
print("Dataset Created Successfully!")
print("="*60)
print()
print(f"Train CSV: {train_csv}")
print(f"Val CSV: {val_csv}")
print()
print("Next steps:")
print("  1. Train model:")
print("     python src/train.py --model_name simple3d --epochs 20 --batch_size 4")
print()
print("  2. Run inference:")
print("     python src/infer_video.py --video test.mp4 --checkpoint results/checkpoints/best_acc.pth --model_name simple3d --class_names normal suspicious theft fighting weapon_gun weapon_other")
print()
print("Note: This is dummy data. For real detection, collect actual videos of:")
print("  - Fighting: CCTV-Fights, RWF-2000 datasets")
print("  - Weapons: Custom security footage, annotated datasets")
print()
