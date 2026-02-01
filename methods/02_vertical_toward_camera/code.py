#!/usr/bin/env python3
"""
Vertical flames toward camera - NO SPARKS (removes black dots).
Original: realistic_v3.py
"""
import cv2
import numpy as np
from ultralytics import YOLO
import random
import math

def create_noise_texture(width, height, scale=30, seed=42):
    """Create a noise texture using numpy."""
    np.random.seed(seed)
    noise = np.random.rand(height // scale + 2, width // scale + 2)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    detail = np.random.rand(height // (scale//2) + 2, width // (scale//2) + 2)
    detail = cv2.resize(detail, (width, height), interpolation=cv2.INTER_CUBIC)
    return 0.7 * noise + 0.3 * detail

def create_turbulent_flame_vertical(width, height, time_offset=0, seed=42, intensity=1.0):
    """Create flame that expands DOWNWARD (toward camera for rear exhaust)."""
    np.random.seed(seed + int(time_offset * 100))
    
    # Multiple turbulence layers
    turb1 = create_noise_texture(width, height, scale=35, seed=seed + int(time_offset*10))
    turb2 = create_noise_texture(width, height, scale=18, seed=seed + int(time_offset*15) + 1000)
    turb3 = create_noise_texture(width, height, scale=9, seed=seed + int(time_offset*20) + 2000)
    
    turbulence = 0.5 * turb1 + 0.35 * turb2 + 0.15 * turb3
    
    # Create vertical flame shape (strongest at top, fading downward)
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    center_x = width / 2
    
    # Vertical gradient (strongest at top, fading down) - flames shoot toward camera
    y_factor = 1 - (y_coords / height)  # 1 at top, 0 at bottom
    y_factor = np.power(y_factor, 0.3)  # Soft falloff
    
    # Horizontal gaussian (concentrated at center)
    x_dist = np.abs(x_coords - center_x) / (width / 2)
    x_factor = np.exp(-2 * x_dist ** 2)
    
    # Add some spread as flame goes down
    spread_factor = 1 + 0.8 * (y_coords / height)  # Wider at bottom
    x_factor_spread = np.exp(-2 * (x_dist / spread_factor) ** 2)
    
    # Combine
    flame_shape = turbulence * y_factor * x_factor_spread
    flame_shape = np.clip(flame_shape * 2.5 * intensity - 0.25, 0, 1)
    
    # Irregular edges
    edge_turb = create_noise_texture(width, height, scale=12, seed=seed + int(time_offset*25) + 3000)
    flame_shape = flame_shape * (0.4 + 0.6 * edge_turb)
    
    return np.clip(flame_shape, 0, 1).astype(np.float32)

def colorize_flame(flame_shape):
    """Convert to BGR with proper temperature gradient."""
    height, width = flame_shape.shape
    flame_bgr = np.zeros((height, width, 3), dtype=np.float32)
    val = flame_shape
    
    # White hot core (> 0.75)
    mask = val > 0.75
    flame_bgr[mask] = [255, 255, 255]
    
    # Bright yellow (0.55 - 0.75)
    mask = (val > 0.55) & (val <= 0.75)
    t = (val[mask] - 0.55) / 0.2
    flame_bgr[mask, 0] = 150 + 105 * t  # B
    flame_bgr[mask, 1] = 255  # G
    flame_bgr[mask, 2] = 255  # R
    
    # Yellow-orange (0.35 - 0.55)
    mask = (val > 0.35) & (val <= 0.55)
    t = (val[mask] - 0.35) / 0.2
    flame_bgr[mask, 0] = 30 + 120 * t  # B
    flame_bgr[mask, 1] = 180 + 75 * t  # G
    flame_bgr[mask, 2] = 255  # R
    
    # Orange (0.18 - 0.35)
    mask = (val > 0.18) & (val <= 0.35)
    t = (val[mask] - 0.18) / 0.17
    flame_bgr[mask, 0] = 0 + 30 * t  # B
    flame_bgr[mask, 1] = 100 + 80 * t  # G
    flame_bgr[mask, 2] = 220 + 35 * t  # R
    
    # Red-orange (0.05 - 0.18)
    mask = (val > 0.05) & (val <= 0.18)
    t = (val[mask] - 0.05) / 0.13
    flame_bgr[mask, 0] = 0
    flame_bgr[mask, 1] = 40 + 60 * t  # G
    flame_bgr[mask, 2] = 150 + 70 * t  # R
    
    # Dark red fade (0 - 0.05)
    mask = (val > 0) & (val <= 0.05)
    t = val[mask] / 0.05
    flame_bgr[mask, 0] = 0
    flame_bgr[mask, 1] = t * 40
    flame_bgr[mask, 2] = 80 + 70 * t
    
    return np.clip(flame_bgr, 0, 255).astype(np.uint8)

def add_bloom(flame_img, strength=0.7):
    """Add glow effect."""
    blurred = cv2.GaussianBlur(flame_img, (51, 51), 0)
    return np.clip(cv2.addWeighted(flame_img, 1.0, blurred, strength, 0), 0, 255).astype(np.uint8)

def screen_blend(background, flame, x, y, alpha_mask):
    """Screen blend for bright daylight."""
    h, w = flame.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    x1_bg = max(0, x)
    y1_bg = max(0, y)
    x2_bg = min(bg_w, x + w)
    y2_bg = min(bg_h, y + h)
    
    x1_fl = max(0, -x)
    y1_fl = max(0, -y)
    x2_fl = x1_fl + (x2_bg - x1_bg)
    y2_fl = y1_fl + (y2_bg - y1_bg)
    
    if x2_bg <= x1_bg or y2_bg <= y1_bg:
        return background
    
    bg_roi = background[y1_bg:y2_bg, x1_bg:x2_bg].astype(np.float32) / 255.0
    fl_roi = flame[y1_fl:y2_fl, x1_fl:x2_fl].astype(np.float32) / 255.0
    alpha = alpha_mask[y1_fl:y2_fl, x1_fl:x2_fl, np.newaxis]
    
    screen = 1 - (1 - fl_roi) * (1 - bg_roi)
    blended = bg_roi * (1 - alpha) + screen * alpha
    
    result = background.copy()
    result[y1_bg:y2_bg, x1_bg:x2_bg] = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return result

def main():
    video_path = "/tmp/video_editing_problem/Video Editing问题数据样例/1105_raw.mp4"
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}, {fps}fps")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/tmp/flame_fix/fixed_vertical_nodots.mp4', fourcc, fps, (width, height))
    
    print("Loading YOLO...")
    model = YOLO('yolov8n.pt')
    
    smooth_bbox = None
    
    # Flame size - taller than wide for vertical projection
    flame_w, flame_h = 140, 200
    
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
                smooth_bbox = [0.85 * s + 0.15 * c for s, c in zip(smooth_bbox, car_bbox)]
        
        if smooth_bbox:
            x1, y1, x2, y2 = smooth_bbox
            car_w = x2 - x1
            car_h = y2 - y1
            car_cx = (x1 + x2) / 2
            car_bottom = y2
            
            # Exhaust positions
            exhaust_l_x = int(car_cx - car_w * 0.15)
            exhaust_r_x = int(car_cx + car_w * 0.15)
            exhaust_y = int(car_bottom - car_h * 0.02)  # Very close to bottom
            
            time_offset = frame_idx * 0.18
            
            # Flickering
            intensity_l = 0.65 + 0.35 * (0.5 + 0.5 * math.sin(frame_idx * 0.35))
            intensity_r = 0.65 + 0.35 * (0.5 + 0.5 * math.sin(frame_idx * 0.45 + 1.3))
            
            # Generate vertical flames
            flame_shape_l = create_turbulent_flame_vertical(flame_w, flame_h, time_offset, seed=42, intensity=intensity_l)
            flame_shape_r = create_turbulent_flame_vertical(flame_w, flame_h, time_offset + 7, seed=123, intensity=intensity_r)
            
            flame_l = colorize_flame(flame_shape_l)
            flame_r = colorize_flame(flame_shape_r)
            
            flame_l = add_bloom(flame_l, 0.75)
            flame_r = add_bloom(flame_r, 0.75)
            
            # Position - flame origin at top, expanding downward
            pos_l_x = exhaust_l_x - flame_w // 2
            pos_l_y = exhaust_y - 15  # Slight overlap with car
            pos_r_x = exhaust_r_x - flame_w // 2
            pos_r_y = exhaust_y - 15
            
            frame = screen_blend(frame, flame_l, pos_l_x, pos_l_y, flame_shape_l)
            frame = screen_blend(frame, flame_r, pos_r_x, pos_r_y, flame_shape_r)
            
            # NO SPARKS - removed to eliminate black dots
        
        out.write(frame)
        
        if frame_idx in [0, 25, 50, 75, 100]:
            cv2.imwrite(f'/tmp/flame_fix/fixed_vertical_nodots_f{frame_idx}.jpg', frame)
            print(f"Frame {frame_idx}")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print("Converting to h264...")
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', '/tmp/flame_fix/fixed_vertical_nodots.mp4',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '/tmp/flame_fix/fixed_vertical_nodots_h264.mp4'
    ], capture_output=True)
    
    print("Done: /tmp/flame_fix/fixed_vertical_nodots_h264.mp4")

if __name__ == '__main__':
    main()
