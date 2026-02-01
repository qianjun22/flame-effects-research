#!/usr/bin/env python3
"""
Flames blowing TOWARD CAMERA (viewer perspective) - NO SPARKS.
Flames appear to jet out at the viewer from the car exhaust.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import math

def create_noise_texture(width, height, scale=30, seed=42):
    """Create turbulence noise."""
    np.random.seed(seed)
    noise = np.random.rand(height // scale + 2, width // scale + 2)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    detail = np.random.rand(height // (scale//2) + 2, width // (scale//2) + 2)
    detail = cv2.resize(detail, (width, height), interpolation=cv2.INTER_CUBIC)
    return 0.7 * noise + 0.3 * detail

def create_camera_flame(width, height, time_offset=0, seed=42, intensity=1.0):
    """
    Create flame that appears to shoot TOWARD the camera.
    - Origin at TOP-CENTER (exhaust pipe position)
    - Expands DOWNWARD and OUTWARD (perspective: closer = larger)
    - Turbulent, irregular edges
    """
    np.random.seed(seed + int(time_offset * 100))
    
    # Multiple turbulence layers for realistic look
    turb1 = create_noise_texture(width, height, scale=25, seed=seed + int(time_offset*10))
    turb2 = create_noise_texture(width, height, scale=12, seed=seed + int(time_offset*15) + 1000)
    turb3 = create_noise_texture(width, height, scale=6, seed=seed + int(time_offset*20) + 2000)
    
    turbulence = 0.45 * turb1 + 0.35 * turb2 + 0.20 * turb3
    
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    center_x = width / 2
    
    # Vertical factor: strongest at TOP (origin), fading as it comes toward camera (down)
    y_norm = y_coords / height  # 0 at top, 1 at bottom
    y_factor = np.power(1 - y_norm, 0.5)  # Fade as it extends toward camera
    
    # Horizontal spread increases as flame comes toward camera (perspective)
    spread = 0.3 + 1.5 * y_norm  # Narrow at top (exhaust), wide at bottom (toward viewer)
    x_dist = np.abs(x_coords - center_x) / (width / 2)
    x_factor = np.exp(-3 * (x_dist / spread) ** 2)
    
    # Core shape with perspective expansion
    flame_shape = turbulence * y_factor * x_factor
    flame_shape = np.clip(flame_shape * 3.0 * intensity - 0.2, 0, 1)
    
    # Add irregular torn edges
    edge_turb = create_noise_texture(width, height, scale=10, seed=seed + int(time_offset*25) + 3000)
    flame_shape = flame_shape * (0.35 + 0.65 * edge_turb)
    
    # Add some extra wisps at the edges
    wisp_turb = create_noise_texture(width, height, scale=8, seed=seed + int(time_offset*30) + 4000)
    wisps = np.clip(wisp_turb * y_factor * 0.7 - 0.5, 0, 1) * 0.4
    flame_shape = np.clip(flame_shape + wisps * (1 - flame_shape), 0, 1)
    
    return np.clip(flame_shape, 0, 1).astype(np.float32)

def colorize_flame(flame_shape):
    """Temperature-based coloring: white core → yellow → orange → red."""
    height, width = flame_shape.shape
    flame_bgr = np.zeros((height, width, 3), dtype=np.float32)
    val = flame_shape
    
    # White hot core (> 0.8)
    mask = val > 0.8
    flame_bgr[mask] = [255, 255, 255]
    
    # Bright yellow (0.6 - 0.8)
    mask = (val > 0.6) & (val <= 0.8)
    t = (val[mask] - 0.6) / 0.2
    flame_bgr[mask, 0] = 100 + 155 * t  # B
    flame_bgr[mask, 1] = 255  # G
    flame_bgr[mask, 2] = 255  # R
    
    # Yellow-orange (0.4 - 0.6)
    mask = (val > 0.4) & (val <= 0.6)
    t = (val[mask] - 0.4) / 0.2
    flame_bgr[mask, 0] = 20 + 80 * t  # B
    flame_bgr[mask, 1] = 170 + 85 * t  # G
    flame_bgr[mask, 2] = 255  # R
    
    # Orange (0.2 - 0.4)
    mask = (val > 0.2) & (val <= 0.4)
    t = (val[mask] - 0.2) / 0.2
    flame_bgr[mask, 0] = 0 + 20 * t  # B
    flame_bgr[mask, 1] = 90 + 80 * t  # G
    flame_bgr[mask, 2] = 200 + 55 * t  # R
    
    # Red-orange (0.08 - 0.2)
    mask = (val > 0.08) & (val <= 0.2)
    t = (val[mask] - 0.08) / 0.12
    flame_bgr[mask, 0] = 0
    flame_bgr[mask, 1] = 30 + 60 * t  # G
    flame_bgr[mask, 2] = 140 + 60 * t  # R
    
    # Dark red fade (0.02 - 0.08)
    mask = (val > 0.02) & (val <= 0.08)
    t = (val[mask] - 0.02) / 0.06
    flame_bgr[mask, 0] = 0
    flame_bgr[mask, 1] = t * 30
    flame_bgr[mask, 2] = 70 + 70 * t
    
    return np.clip(flame_bgr, 0, 255).astype(np.uint8)

def add_bloom(flame_img, strength=0.6):
    """Add glow/bloom effect."""
    blurred = cv2.GaussianBlur(flame_img, (41, 41), 0)
    return np.clip(cv2.addWeighted(flame_img, 1.0, blurred, strength, 0), 0, 255).astype(np.uint8)

def screen_blend(background, flame, x, y, alpha_mask):
    """Screen blend for natural flame integration."""
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
    
    # Screen blend for bright flames
    screen = 1 - (1 - fl_roi) * (1 - bg_roi)
    blended = bg_roi * (1 - alpha * 0.85) + screen * alpha * 0.85
    
    result = background.copy()
    result[y1_bg:y2_bg, x1_bg:x2_bg] = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return result

def main():
    video_path = "/tmp/video_editing_problem/Video Editing问题数据样例/1105_raw.mp4"
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps}fps, {total} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/tmp/flame_fix/fixed_camera.mp4', fourcc, fps, (width, height))
    
    print("Loading YOLO...")
    model = YOLO('yolov8n.pt')
    
    smooth_bbox = None
    
    # Flame dimensions - taller for toward-camera perspective
    flame_w, flame_h = 120, 180
    
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
            
            # Two exhaust positions (left and right of center)
            exhaust_l_x = int(car_cx - car_w * 0.13)
            exhaust_r_x = int(car_cx + car_w * 0.13)
            exhaust_y = int(car_bottom - car_h * 0.03)
            
            time_offset = frame_idx * 0.2
            
            # Dynamic flickering (different phase for each flame)
            intensity_l = 0.7 + 0.3 * (0.5 + 0.5 * math.sin(frame_idx * 0.38))
            intensity_r = 0.7 + 0.3 * (0.5 + 0.5 * math.sin(frame_idx * 0.42 + 1.5))
            
            # Size variation
            size_var_l = 0.85 + 0.3 * math.sin(frame_idx * 0.25 + 0.5)
            size_var_r = 0.85 + 0.3 * math.sin(frame_idx * 0.28 + 2.0)
            
            fl_w_l = int(flame_w * size_var_l)
            fl_h_l = int(flame_h * size_var_l)
            fl_w_r = int(flame_w * size_var_r)
            fl_h_r = int(flame_h * size_var_r)
            
            # Generate flames shooting toward camera
            flame_shape_l = create_camera_flame(fl_w_l, fl_h_l, time_offset, seed=42, intensity=intensity_l)
            flame_shape_r = create_camera_flame(fl_w_r, fl_h_r, time_offset + 5, seed=999, intensity=intensity_r)
            
            flame_l = colorize_flame(flame_shape_l)
            flame_r = colorize_flame(flame_shape_r)
            
            flame_l = add_bloom(flame_l, 0.65)
            flame_r = add_bloom(flame_r, 0.65)
            
            # Position flames - origin (top of flame) at exhaust, expanding downward toward viewer
            pos_l_x = exhaust_l_x - fl_w_l // 2
            pos_l_y = exhaust_y - 10  # Slight overlap with car
            pos_r_x = exhaust_r_x - fl_w_r // 2
            pos_r_y = exhaust_y - 10
            
            frame = screen_blend(frame, flame_l, pos_l_x, pos_l_y, flame_shape_l)
            frame = screen_blend(frame, flame_r, pos_r_x, pos_r_y, flame_shape_r)
        
        out.write(frame)
        
        if frame_idx in [0, 30, 60, 90, 120]:
            cv2.imwrite(f'/tmp/flame_fix/camera_f{frame_idx}.jpg', frame)
            print(f"Frame {frame_idx}")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print("Converting to h264...")
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', '/tmp/flame_fix/fixed_camera.mp4',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '/tmp/flame_fix/fixed_camera_h264.mp4'
    ], capture_output=True)
    
    print("Done: /tmp/flame_fix/fixed_camera_h264.mp4")

if __name__ == '__main__':
    main()
