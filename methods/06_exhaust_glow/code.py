#!/usr/bin/env python3
"""
Exhaust flames with glowing pipe interior - based on close-up reference.
Bright core inside pipe, flames shooting out, radial glow, motion blur.
"""
import cv2
import numpy as np
from ultralytics import YOLO

RAW = "/tmp/video_editing_problem/Video Editing问题数据样例/1105_raw.mp4"
OUT = "/tmp/flame_fix/fixed_glow"

# Proper fire colors (BGR)
WHITE_CORE = np.array([255, 255, 255], dtype=np.float32)
YELLOW = np.array([0, 255, 255], dtype=np.float32)
ORANGE = np.array([0, 165, 255], dtype=np.float32)
DEEP_ORANGE = np.array([0, 100, 255], dtype=np.float32)
RED = np.array([0, 50, 200], dtype=np.float32)

def create_exhaust_glow(length, height, intensity, seed):
    """Create exhaust with glowing pipe interior and flames shooting out."""
    np.random.seed(seed)
    
    pad = 60
    w = length + 2 * pad
    h = height + 2 * pad
    
    flame = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    
    cy = h // 2
    pipe_x = w - pad
    
    # 1. GLOWING PIPE INTERIOR - bright radial gradient at source
    # This simulates looking into a glowing hot pipe
    inner_r = int(height * 0.25)  # Pipe opening radius
    outer_glow_r = int(height * 0.5)  # Glow extends beyond
    
    for dx in range(-outer_glow_r, outer_glow_r + 1):
        for dy in range(-outer_glow_r, outer_glow_r + 1):
            dist = np.sqrt(dx*dx + dy*dy)
            px, py = pipe_x + dx, cy + dy
            
            if 0 <= px < w and 0 <= py < h:
                if dist <= inner_r:
                    # Inside pipe - very bright white/yellow core
                    t = dist / inner_r
                    if t < 0.3:
                        color = WHITE_CORE
                        a = 1.0
                    elif t < 0.6:
                        color = WHITE_CORE * 0.7 + YELLOW * 0.3
                        a = 0.95
                    else:
                        color = YELLOW * 0.8 + ORANGE * 0.2
                        a = 0.9
                elif dist <= outer_glow_r:
                    # Radial glow around pipe opening
                    t = (dist - inner_r) / (outer_glow_r - inner_r)
                    color = ORANGE * (1-t) + DEEP_ORANGE * t
                    a = (1 - t) * 0.6
                else:
                    continue
                
                a *= intensity
                flame[py, px] = np.clip(flame[py, px] + color * a, 0, 255)
                alpha[py, px] = min(1.0, alpha[py, px] + a)
    
    # 2. FLAMES SHOOTING OUT from the glowing center
    n_streaks = np.random.randint(8, 14)
    
    for si in range(n_streaks):
        # Start from within the pipe opening
        start_r = np.random.uniform(0, inner_r * 0.7)
        start_angle = np.random.uniform(-0.5, 0.5)
        start_y = cy + int(start_r * np.sin(start_angle * 3))
        
        streak_len = int(length * np.random.uniform(0.4, 0.9))
        streak_width = np.random.randint(4, 12)
        
        # Slight angle variation
        angle = np.random.uniform(-0.15, 0.15)
        
        # Turbulence
        wave_freq = np.random.uniform(0.12, 0.25)
        wave_amp = np.random.uniform(3, 8)
        
        for i in range(streak_len):
            progress = i / streak_len
            x = pipe_x - i
            
            wave = wave_amp * np.sin(i * wave_freq + si) * (0.5 + progress)
            y_center = start_y + int(angle * i + wave)
            
            # Width tapers
            local_w = int(streak_width * (1 - progress * 0.55))
            
            for dy in range(-local_w, local_w + 1):
                y = y_center + dy
                
                if np.random.random() > 0.93:
                    continue
                
                if 0 <= y < h and 0 <= x < w:
                    d = abs(dy) / (local_w + 0.1)
                    
                    # Color based on distance from source
                    if progress < 0.15:
                        color = YELLOW * (1-d) + ORANGE * d
                    elif progress < 0.4:
                        t = (progress - 0.15) / 0.25
                        base = YELLOW * (1-t) + ORANGE * t
                        color = base * (1-d*0.5) + DEEP_ORANGE * (d*0.5)
                    elif progress < 0.7:
                        t = (progress - 0.4) / 0.3
                        color = ORANGE * (1-t-d*0.3) + DEEP_ORANGE * (t+d*0.3)
                    else:
                        color = DEEP_ORANGE * (1-d*0.5) + RED * (d*0.5)
                    
                    a = (1 - d * 0.5) * (1 - progress * 0.45) * intensity
                    a *= (0.75 + 0.25 * np.random.random())
                    a = min(1.0, max(0, a))
                    
                    if a > 0.02:
                        flame[y, x] = np.clip(flame[y, x] + color * a * 0.45, 0, 255)
                        alpha[y, x] = min(1.0, alpha[y, x] + a * 0.4)
    
    # 3. Add bloom/glow effect by blurring and adding back
    glow = cv2.GaussianBlur(flame, (31, 31), 0)
    flame = np.clip(flame + glow * 0.4, 0, 255)
    
    glow_alpha = cv2.GaussianBlur(alpha, (25, 25), 0)
    alpha = np.clip(alpha + glow_alpha * 0.3, 0, 1)
    
    # 4. Strong horizontal motion blur
    kernel = np.zeros((5, 21))
    kernel[2, :] = 1.0 / 21
    flame = cv2.filter2D(flame, -1, kernel)
    alpha = cv2.filter2D(alpha, -1, kernel)
    
    # Light final blur
    flame = cv2.GaussianBlur(flame, (5, 5), 0)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    return flame.astype(np.uint8), alpha

def paste_flame(frame, flame_bgr, flame_alpha, x, y, flip=False):
    if flip:
        flame_bgr = cv2.flip(flame_bgr, 1)
        flame_alpha = cv2.flip(flame_alpha, 1)
    
    fh, fw = flame_bgr.shape[:2]
    H, W = frame.shape[:2]
    
    if flip:
        tx = x - 60
    else:
        tx = x - fw + 60
    ty = y - fh // 2
    
    src_x1, src_y1 = max(0, -tx), max(0, -ty)
    dst_x1, dst_y1 = max(0, tx), max(0, ty)
    src_x2 = min(fw, W - tx)
    src_y2 = min(fh, H - ty)
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return frame
    
    fg = flame_bgr[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
    a = flame_alpha[src_y1:src_y2, src_x1:src_x2]
    
    result = frame.copy()
    bg = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    
    # Screen blend for bright glow
    for c in range(3):
        blended = bg[:,:,c] + fg[:,:,c] * a - (bg[:,:,c] * fg[:,:,c] * a) / 255
        result[dst_y1:dst_y2, dst_x1:dst_x2, c] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return result

def main():
    print("Loading...")
    cap = cv2.VideoCapture(RAW)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    model = YOLO('yolov8n.pt')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT + '_temp.mp4', fourcc, fps, (w, h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    results = model(frame0, classes=[2], verbose=False)
    
    ref_car = (888, 407, 1159, 595)
    for r in results:
        for box in r.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            if (bx2 - bx1) > 150:
                ref_car = (bx1, by1, bx2, by2)
                break
    
    car_w = ref_car[2] - ref_car[0]
    left_off = (int(-car_w * 0.26), 0)
    right_off = (int(car_w * 0.26), 0)
    
    print("Processing...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_car = ref_car
    
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, classes=[2], verbose=False)
        car_box = None
        for r in results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                if (bx2 - bx1) > 150:
                    car_box = (bx1, by1, bx2, by2)
                    break
        
        if car_box is None:
            car_box = prev_car
        else:
            car_box = tuple(int(0.85 * p + 0.15 * c) for c, p in zip(car_box, prev_car))
        prev_car = car_box
        
        car_cx = (car_box[0] + car_box[2]) // 2
        car_bottom = car_box[3]
        
        t = i * 0.15
        left_len = 120 + int(35 * np.sin(t * 1.0))
        left_h = 50 + int(12 * np.sin(t * 0.75 + 1))
        right_len = 140 + int(45 * np.sin(t * 0.85 + 2))
        right_h = 45 + int(15 * np.sin(t * 0.9 + 1.5))
        
        intensity = 0.85 + 0.25 * np.sin(t * 0.5)
        
        left_flame, left_alpha = create_exhaust_glow(left_len, left_h, intensity, i * 61)
        right_flame, right_alpha = create_exhaust_glow(right_len, right_h, intensity, i * 61 + 9000)
        
        output = frame.copy()
        output = paste_flame(output, left_flame, left_alpha, 
                            car_cx + left_off[0], car_bottom + left_off[1], flip=True)
        output = paste_flame(output, right_flame, right_alpha,
                            car_cx + right_off[0], car_bottom + right_off[1], flip=False)
        
        out.write(output)
        
        if i % 25 == 0:
            cv2.imwrite(f"{OUT}_f{i}.jpg", output)
            print(f"  Frame {i}")
    
    cap.release()
    out.release()
    
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', OUT + '_temp.mp4',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        OUT + '_h264.mp4'
    ], capture_output=True)
    
    print(f"Done: {OUT}_h264.mp4")

if __name__ == "__main__":
    main()
