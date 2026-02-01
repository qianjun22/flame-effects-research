#!/usr/bin/env python3
"""
Cloud-like explosive flames - billowing fire clouds, not beams.
Based on reference images showing large turbulent flame CLOUDS.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import random
import math

def turbulent_cloud(width, height, time=0, seed=42):
    """Generate turbulent cloud-like texture."""
    np.random.seed(seed + int(time * 40))
    
    # Multiple noise octaves
    result = np.zeros((height, width), dtype=np.float32)
    
    for i, (scale, weight) in enumerate([(60, 0.35), (30, 0.30), (15, 0.20), (8, 0.15)]):
        noise = np.random.rand(height // scale + 2, width // scale + 2)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Animated distortion
        dx = int(8 * math.sin(time * (i+1) * 0.4 + i))
        dy = int(8 * math.cos(time * (i+1) * 0.5 + i*0.7))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        noise = cv2.warpAffine(noise, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        
        result += weight * noise
    
    return result

def create_flame_cloud(width, height, time=0, seed=42, intensity=1.0):
    """Create a cloud-like flame shape (blob, not jet)."""
    turb = turbulent_cloud(width, height, time, seed)
    
    # Create circular/blob base shape (not cone)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    cx, cy = width / 2, height / 2
    
    # Elliptical shape, wider than tall (like a horizontal cloud)
    dist_x = (x_coords - cx) / (width / 2)
    dist_y = (y_coords - cy) / (height / 2)
    
    # Ellipse with turbulent edges
    ellipse = np.exp(-2 * (dist_x**2 + dist_y**2 * 1.5))
    
    # Add turbulent distortion to the shape
    turb2 = turbulent_cloud(width, height, time * 1.3, seed + 1000)
    
    # Combine
    shape = ellipse * turb * (0.5 + 0.5 * turb2)
    
    # Boost the core
    shape = np.clip(shape * 3.0 * intensity - 0.1, 0, 1)
    
    # Power curve to enhance core brightness
    shape = np.power(shape, 0.7)
    
    return np.clip(shape, 0, 1).astype(np.float32)

def colorize_realistic(shape):
    """Realistic flame colors with strong white core."""
    h, w = shape.shape
    result = np.zeros((h, w, 3), dtype=np.float32)
    val = shape
    
    # Pure white overexposed core (>0.8)
    mask = val > 0.8
    result[mask] = [255, 255, 255]
    
    # White-yellow (0.65-0.8)
    mask = (val > 0.65) & (val <= 0.8)
    t = (val[mask] - 0.65) / 0.15
    result[mask, 0] = 200 + 55 * t
    result[mask, 1] = 255
    result[mask, 2] = 255
    
    # Bright yellow (0.5-0.65)
    mask = (val > 0.5) & (val <= 0.65)
    t = (val[mask] - 0.5) / 0.15
    result[mask, 0] = 100 + 100 * t
    result[mask, 1] = 240 + 15 * t
    result[mask, 2] = 255
    
    # Yellow-orange (0.35-0.5)
    mask = (val > 0.35) & (val <= 0.5)
    t = (val[mask] - 0.35) / 0.15
    result[mask, 0] = 40 + 60 * t
    result[mask, 1] = 180 + 60 * t
    result[mask, 2] = 255
    
    # Orange (0.2-0.35)
    mask = (val > 0.2) & (val <= 0.35)
    t = (val[mask] - 0.2) / 0.15
    result[mask, 0] = 0 + 40 * t
    result[mask, 1] = 120 + 60 * t
    result[mask, 2] = 220 + 35 * t
    
    # Red-orange (0.08-0.2)
    mask = (val > 0.08) & (val <= 0.2)
    t = (val[mask] - 0.08) / 0.12
    result[mask, 0] = 0
    result[mask, 1] = 60 + 60 * t
    result[mask, 2] = 160 + 60 * t
    
    # Dark red fade (0-0.08)
    mask = (val > 0) & (val <= 0.08)
    t = val[mask] / 0.08
    result[mask, 0] = 0
    result[mask, 1] = t * 60
    result[mask, 2] = 100 + 60 * t
    
    return np.clip(result, 0, 255).astype(np.uint8)

def add_glow(img):
    """Heavy glow for fire."""
    b1 = cv2.GaussianBlur(img, (81, 81), 0)
    b2 = cv2.GaussianBlur(img, (41, 41), 0)
    result = img.astype(np.float32) + 0.5 * b1 + 0.3 * b2
    return np.clip(result, 0, 255).astype(np.uint8)

class Spark:
    def __init__(self, x, y):
        angle = random.uniform(0, 2*math.pi)  # All directions
        speed = random.uniform(5, 20)
        self.x = x + random.gauss(0, 15)
        self.y = y + random.gauss(0, 10)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle) * 0.6
        self.lifetime = random.randint(10, 30)
        self.age = 0
        self.size = random.choice([1, 1, 2, 2, 3])
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.25  # gravity
        self.vx *= 0.95
        self.age += 1
    
    def draw(self, img):
        if self.age >= self.lifetime:
            return
        x, y = int(self.x), int(self.y)
        h, w = img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            fade = (1 - self.age / self.lifetime) ** 0.7
            color = (int(80*fade), int(220*fade), int(255*fade))
            cv2.circle(img, (x, y), self.size, color, -1)

def blend_additive(bg, flame, alpha, x, y):
    """Additive blend."""
    h, w = flame.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    x, y = int(x), int(y)
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x+w), min(bg_h, y+h)
    fx1, fy1 = max(0, -x), max(0, -y)
    fx2, fy2 = fx1+(x2-x1), fy1+(y2-y1)
    
    if x2 <= x1 or y2 <= y1:
        return bg
    
    result = bg.copy()
    al = alpha[fy1:fy2, fx1:fx2, np.newaxis]
    blended = result[y1:y2, x1:x2].astype(np.float32) + flame[fy1:fy2, fx1:fx2].astype(np.float32) * al
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result

def main():
    cap = cv2.VideoCapture("/tmp/video_editing_problem/Video Editing问题数据样例/1105_raw.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/tmp/flame_fix/fixed_cloud.mp4', fourcc, fps, (width, height))
    
    print("Loading YOLO...")
    model = YOLO('yolov8n.pt')
    
    sparks = []
    smooth_bbox = None
    
    # Large cloud-like flames (wider than tall)
    flame_w, flame_h = 350, 250
    
    print("Processing...")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, classes=[2], verbose=False)
        
        car_bbox = None
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if (x2-x1) * (y2-y1) > 10000:
                    car_bbox = [x1, y1, x2, y2]
                    break
        
        if car_bbox:
            if smooth_bbox is None:
                smooth_bbox = car_bbox
            else:
                smooth_bbox = [0.85*s + 0.15*c for s, c in zip(smooth_bbox, car_bbox)]
        
        if smooth_bbox:
            x1, y1, x2, y2 = smooth_bbox
            car_w = x2 - x1
            car_h = y2 - y1
            car_cx = (x1 + x2) / 2
            car_bottom = y2
            
            # Position flame cloud at exhaust area
            flame_cx = int(car_cx)
            flame_cy = int(car_bottom + flame_h * 0.2)  # Slightly below car
            
            time = frame_idx * 0.25
            
            # Pulsing intensity
            intensity = 0.75 + 0.25 * math.sin(frame_idx * 0.5)
            # Random bursts
            if random.random() < 0.15:
                intensity *= 1.3
            
            # Create flame cloud
            shape = create_flame_cloud(flame_w, flame_h, time=time, seed=42, intensity=intensity)
            flame = colorize_realistic(shape)
            flame = add_glow(flame)
            
            # Center flame cloud
            pos_x = flame_cx - flame_w // 2
            pos_y = flame_cy - flame_h // 2
            
            frame = blend_additive(frame, flame, shape, pos_x, pos_y)
            
            # Sparks flying in all directions
            if random.random() < 0.6:
                sparks.extend([Spark(flame_cx, flame_cy - flame_h//4) for _ in range(random.randint(5, 15))])
            
            for spark in sparks[:]:
                spark.update()
                spark.draw(frame)
                if spark.age >= spark.lifetime:
                    sparks.remove(spark)
        
        out.write(frame)
        
        if frame_idx in [0, 25, 50, 75, 100]:
            cv2.imwrite(f'/tmp/flame_fix/fixed_cloud_f{frame_idx}.jpg', frame)
            print(f"Frame {frame_idx}")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print("Converting...")
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', '/tmp/flame_fix/fixed_cloud.mp4',
                   '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                   '/tmp/flame_fix/fixed_cloud_h264.mp4'], capture_output=True)
    print("Done!")

if __name__ == '__main__':
    main()
