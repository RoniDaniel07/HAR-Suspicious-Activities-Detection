"""
Create a simple test video for inference testing
"""
import cv2
import numpy as np

print("Creating test video...")

# Video parameters
width, height = 640, 480
fps = 10
duration = 10  # seconds
total_frames = fps * duration

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))

# Generate frames
for i in range(total_frames):
    # Create a frame with some movement
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles that move
    x = int((i / total_frames) * width)
    y = height // 2
    
    # Background
    frame[:] = (50, 50, 50)
    
    # Moving rectangle (simulating a person)
    cv2.rectangle(frame, (x, y-50), (x+100, y+50), (0, 255, 0), -1)
    
    # Add some text
    cv2.putText(frame, f'Frame {i+1}/{total_frames}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add timestamp
    time = i / fps
    cv2.putText(frame, f'Time: {time:.1f}s', (10, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(frame)

out.release()

print(f"✓ Test video created: test_video.mp4")
print(f"  Duration: {duration} seconds")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total frames: {total_frames}")
print("\nYou can now use this video for inference:")
print("  python src/infer_video.py --video test_video.mp4 --checkpoint results/checkpoints/best_acc.pth --model_name simple3d --output_video output.mp4")
