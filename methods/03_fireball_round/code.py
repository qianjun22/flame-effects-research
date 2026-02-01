#!/usr/bin/env python3
"""
Simple fireball flames - smooth, round, realistic exhaust fire.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

RAW_VIDEO = "/tmp/video_editing_problem/Video Editing问题数据样例/1105_raw.mp4"
OUTPUT_VIDEO = "/tmp/flame_fix/fixed_fireball.mp4"
OUTPUT_H264 = "/tmp/flame_fix/fixed_fireball_h264.mp4"

print("Loading YOLOv8...")
model = YOLO('yolov8n.pt')

def detect_car_bbox(model, frame):
    results = model(frame, verbose=False, classes=[2])
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    areas = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        areas.append((xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1]))
    return boxes[np.argmax(areas)].xyxy[0].cpu().numpy()

def estimate_plate_bbox(car_bbox):
    x1, y1, x2, y2 = [int(b) for b in car_bbox]
    car_w = x2 - x1
    car_h = y2 - y1
    plate_w = int(car_w * 0.26)
    plate_h = int(plate_w / 2.5)
    plate_x = x1 + (car_w - plate_w) // 2
    plate_y = y2 - int(car_h * 0.2)
    return (plate_x, plate_y, plate_w, plate_h)

def get_flame_positions(car_bbox, plate_bbox):
    car_x1, car_y1, car_x2, car_y2 = [int(b) for b in car_bbox]
    plate_x, plate_y, plate_w, plate_h = plate_bbox
    left_x = (car_x1 + plate_x) // 2
    right_x = (plate_x + plate_w + car_x2) // 2
    flame_y = plate_y + plate_h
    return (left_x, flame_y), (right_x, flame_y)

def create_fireball(size=80):
    """Create a simple smooth fireball - no artificial patterns."""
    flame = np.zeros((size, size, 4), dtype=np.float32)
    cx, cy = size // 2, size // 2
    
    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx*dx + dy*dy)
            r = size / 2
            
            if dist < r:
                # Smooth radial gradient
                t = dist / r  # 0 at center, 1 at edge
                
                # Alpha falloff - smooth edge
                alpha = (1.0 - t**1.5) ** 2
                
                # Color gradient: white->yellow->orange->red
                if t < 0.3:
                    color = [255, 255, 220]  # white-yellow
                elif t < 0.55:
                    s = (t - 0.3) / 0.25
                    color = [255, int(255 - 45*s), int(220 - 150*s)]  # yellow->orange
                elif t < 0.8:
                    s = (t - 0.55) / 0.25
                    color = [255, int(210 - 85*s), int(70 - 45*s)]  # orange
                else:
                    s = (t - 0.8) / 0.2
                    color = [int(255 - 60*s), int(125 - 70*s), int(25 - 15*s)]  # red
                
                flame[y, x] = [color[2], color[1], color[0], alpha * 255]
    
    # Heavy blur for smooth look
    flame = cv2.GaussianBlur(flame, (15, 15), 0)
    return flame.astype(np.uint8)

def blend_flame(frame, flame, pos, scale=1.0):
    h, w = flame.shape[:2]
    h = int(h * scale)
    w = int(w * scale)
    if h <= 0 or w <= 0:
        return frame
    
    flame_resized = cv2.resize(flame, (w, h))
    
    x, y = pos
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x1 + w, y1 + h
    
    fx1, fy1 = max(0, -x1), max(0, -y1)
    fx2 = w - max(0, x2 - frame.shape[1])
    fy2 = h - max(0, y2 - frame.shape[0])
    
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    flame_part = flame_resized[fy1:fy2, fx1:fx2]
    alpha = flame_part[:, :, 3:4].astype(np.float32) / 255.0
    color = flame_part[:, :, :3].astype(np.float32)
    
    # Additive blend
    blended = roi + color * alpha * 1.8
    blended = np.clip(blended, 0, 255)
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame

def main():
    print("=" * 50)
    print("Fireball Flames")
    print("=" * 50)
    
    cap = cv2.VideoCapture(RAW_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # Detect
    print("Detecting...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    car_bboxes, plate_bboxes = [], []
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        car = detect_car_bbox(model, frame)
        if car is not None:
            plate = estimate_plate_bbox(car)
            car_bboxes.append(car)
            plate_bboxes.append(plate)
        else:
            car_bboxes.append(None)
            plate_bboxes.append(None)
    
    # Create fireball
    print("Creating fireball...")
    fireball = create_fireball(90)
    
    # Process
    print("Adding flames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        car, plate = car_bboxes[i], plate_bboxes[i]
        
        if car is not None and plate is not None:
            # Smooth
            window = 3
            vc = [c for c in car_bboxes[max(0,i-window):min(len(car_bboxes),i+window+1)] if c is not None]
            vp = [p for p in plate_bboxes[max(0,i-window):min(len(plate_bboxes),i+window+1)] if p is not None]
            if vc:
                car = tuple(int(np.mean([c[j] for c in vc])) for j in range(4))
            if vp:
                plate = tuple(int(np.mean([p[j] for p in vp])) for j in range(4))
            
            left, right = get_flame_positions(car, plate)
            
            # Pulsing scale
            scale = 0.95 + 0.15 * np.sin(i * 0.6)
            
            frame = blend_flame(frame, fireball, left, scale)
            frame = blend_flame(frame, fireball, right, scale * 0.97)
            
            if i % 25 == 0:
                print(f"  Frame {i}: L={left}, R={right}")
        
        out.write(frame)
        
        if i in [0, 30, 60, 100]:
            cv2.imwrite(f"/tmp/flame_fix/fireball_f{i}.jpg", frame)
    
    cap.release()
    out.release()
    
    print("Converting to H264...")
    os.system(f'ffmpeg -y -i {OUTPUT_VIDEO} -c:v libx264 -preset fast -crf 18 {OUTPUT_H264} 2>/dev/null')
    
    print(f"Done: {OUTPUT_H264}")

if __name__ == "__main__":
    main()
