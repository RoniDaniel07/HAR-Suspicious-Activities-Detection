"""
Real-time webcam inference for theft/suspicious activity detection
"""
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from collections import deque
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.transforms import VideoTransform
from src.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time webcam inference')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='timesformer')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--class_names', type=str, nargs='+', default=['normal', 'suspicious', 'theft'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--smoothing_window', type=int, default=5, help='Temporal smoothing window')
    parser.add_argument('--inference_interval', type=int, default=10, help='Run inference every N frames')
    
    return parser.parse_args()


class RealtimeDetector:
    """Real-time activity detector"""
    
    def __init__(self, model, transform, args):
        self.model = model
        self.transform = transform
        self.args = args
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=args.num_frames)
        
        # Prediction smoothing
        self.pred_history = deque(maxlen=args.smoothing_window)
        
        # Current prediction
        self.current_pred = 'normal'
        self.current_conf = 0.0
        self.current_probs = None
        
        # Frame counter
        self.frame_count = 0
    
    def add_frame(self, frame):
        """Add frame to buffer"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_buffer.append(frame_rgb)
        self.frame_count += 1
    
    @torch.no_grad()
    def predict(self):
        """Run prediction on current buffer"""
        if len(self.frame_buffer) < self.args.num_frames:
            return
        
        # Get frames
        frames = list(self.frame_buffer)
        
        # Ensure we have exactly num_frames
        if len(frames) > self.args.num_frames:
            # Sample uniformly
            indices = np.linspace(0, len(frames)-1, self.args.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        frames = np.array(frames)
        
        # Transform
        video_tensor = self.transform(frames)
        video_tensor = video_tensor.unsqueeze(0).to(self.args.device)
        
        # Predict
        outputs = self.model(video_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        
        # Add to history for smoothing
        self.pred_history.append(pred_class)
        
        # Smooth prediction (majority vote)
        if len(self.pred_history) >= self.args.smoothing_window:
            smoothed_pred = max(set(self.pred_history), key=self.pred_history.count)
        else:
            smoothed_pred = pred_class
        
        self.current_pred = self.args.class_names[smoothed_pred]
        self.current_conf = probs[smoothed_pred].item()
        self.current_probs = probs.cpu().numpy()
    
    def should_run_inference(self):
        """Check if we should run inference on this frame"""
        return self.frame_count % self.args.inference_interval == 0
    
    def draw_prediction(self, frame):
        """Draw prediction on frame"""
        height, width = frame.shape[:2]
        
        # Choose color based on prediction
        if self.current_pred == 'theft':
            color = (0, 0, 255)  # Red
            bg_color = (0, 0, 128)
        elif self.current_pred == 'suspicious':
            color = (0, 165, 255)  # Orange
            bg_color = (0, 82, 128)
        else:
            color = (0, 255, 0)  # Green
            bg_color = (0, 128, 0)
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (10, 10), (width - 10, 150), color, 3)
        
        # Draw text
        text = f"{self.current_pred.upper()}"
        cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   1.5, (255, 255, 255), 3)
        
        conf_text = f"Confidence: {self.current_conf*100:.1f}%"
        cv2.putText(frame, conf_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)
        
        # Draw probability bars
        if self.current_probs is not None:
            bar_y = 170
            bar_height = 20
            bar_max_width = 200
            
            for i, (class_name, prob) in enumerate(zip(self.args.class_names, self.current_probs)):
                bar_width = int(prob * bar_max_width)
                
                # Draw bar background
                cv2.rectangle(frame, (20, bar_y), (20 + bar_max_width, bar_y + bar_height),
                            (50, 50, 50), -1)
                
                # Draw bar
                if class_name == 'theft':
                    bar_color = (0, 0, 255)
                elif class_name == 'suspicious':
                    bar_color = (0, 165, 255)
                else:
                    bar_color = (0, 255, 0)
                
                cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + bar_height),
                            bar_color, -1)
                
                # Draw label
                label = f"{class_name}: {prob*100:.1f}%"
                cv2.putText(frame, label, (230, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1)
                
                bar_y += bar_height + 10
        
        # Draw FPS
        fps_text = f"FPS: {self.get_fps():.1f}"
        cv2.putText(frame, fps_text, (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        
        # Alert if suspicious/theft detected
        if self.current_pred != 'normal' and self.current_conf >= self.args.threshold:
            alert_text = "!!! ALERT !!!"
            cv2.putText(frame, alert_text, (width // 2 - 150, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        return frame
    
    def get_fps(self):
        """Calculate FPS"""
        if not hasattr(self, 'fps_start_time'):
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
            return 0.0
        
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed > 1.0:
            fps = self.fps_frame_count / elapsed
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
            return fps
        
        return self.fps_frame_count / max(elapsed, 0.001)


def main():
    args = parse_args()
    
    print("Loading model...")
    num_classes = len(args.class_names)
    model = build_model(args.model_name, num_classes=num_classes, pretrained=False)
    load_checkpoint(model, args.checkpoint, device=args.device)
    model = model.to(args.device)
    model.eval()
    
    print("Creating transform...")
    transform = VideoTransform(mode='test')
    
    print("Initializing detector...")
    detector = RealtimeDetector(model, transform, args)
    
    print(f"Opening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera_id}")
        return
    
    print("\nReal-time detection started!")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            # Add frame to detector
            detector.add_frame(frame)
            
            # Run inference periodically
            if detector.should_run_inference():
                detector.predict()
            
            # Draw prediction
            display_frame = detector.draw_prediction(frame)
            
            # Show frame
            cv2.imshow('HAR Theft Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f'screenshot_{screenshot_count:03d}.png'
                cv2.imwrite(screenshot_path, display_frame)
                print(f"Screenshot saved: {screenshot_path}")
                screenshot_count += 1
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == '__main__':
    main()
