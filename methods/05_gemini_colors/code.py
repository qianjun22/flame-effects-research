#!/usr/bin/env python3
"""
Flames using Gemini-recommended BGR colors - no HSV conversion.
Physically accurate: white core → yellow → orange → red tips
"""
import cv2
import numpy as np
from ultralytics import YOLO

RAW = "/tmp/video_editing_problem/Video Editing问题数据样例/1105_raw.mp4"
OUT = "/tmp/flame_fix/fixed_gemini"

# Gemini-recommended BGR colors (hottest to coolest)
COLOR_CORE = np.array([255, 255, 255], dtype=np.float32)    # White (hottest)
COLOR_INNER = np.array([0, 255, 255], dtype=np.float32)     # Yellow
COLOR_MID = np.array([0, 165, 255], dtype=np.float32)       # Orange
COLOR_OUTER = np.array([0, 69, 255], dtype=np.float32)      # Deep orange
COLOR_TIP = np.array([0, 0, 200], dtype=np.float32)         # Red (coolest)

def lerp_color(c1, c2, t):
    """Linear interpolate between two colors."""
    return c1 * (1 - t) + c2 * t

def get_flame_color(progress, dist_from_center):
    """Get flame color based on position. progress=0 at pipe, 1 at tip."""
    # Radial distance affects temperature (center hotter)
    temp = 1 - progress * 0.7 - dist_from_center * 0.3
    temp = max(0, min(1, temp))
    
    if temp > 0.85:  # Hottest - white
        return lerp_color(COLOR_INNER, COLOR_CORE, (temp - 0.85) / 0.15)
    elif temp > 0.65:  # Hot - yellow
        return lerp_color(COLOR_MID, COLOR_INNER, (temp - 0.65) / 0.2)
    elif temp > 0.4:  # Medium - orange
        return lerp_color(COLOR_OUTER, COLOR_MID, (temp - 0.4) / 0.25)
    elif temp > 0.15:  # Cool - deep orange
        return lerp_color(COLOR_TIP, COLOR_OUTER, (temp - 0.15) / 0.25)
    else:  # Coolest - red
        return COLOR_TIP

def create_flame(length, height, intensity, seed):
    """Create realistic flame using proper colors."""
    np.random.seed(seed)
    
    pad = 50
    w = length + 2 * pad
    h = height + 2 * pad
    
    flame = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    
    cy = h // 2
    pipe_x = w - pad  # Pipe at right side
    
    # 1. Bright core at pipe opening
    core_r = int(height * 0.3)
    for dx in range(-core_r, core_r + 1):
        for dy in range(-core_r, core_r + 1):
            dist = np.sqrt(dx*dx + dy*dy)
            if dist <= core_r:
                px, py = pipe_x + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    falloff = dist / core_r
                    color = get_flame_color(0, falloff)
                    a = (1 - falloff * 0.5) * intensity * 1.2
                    flame[py, px] = np.clip(flame[py, px] + color * a, 0, 255)
                    alpha[py, px] = min(1.0, alpha[py, px] + a)
    
    # 2. Flame streaks shooting out
    n_streaks = np.random.randint(10, 16)
    
    for si in range(n_streaks):
        start_y = cy + np.random.randint(-core_r, core_r)
        streak_len = int(length * np.random.uniform(0.35, 0.95))
        streak_width = np.random.randint(3, 10)
        angle = np.random.uniform(-0.2, 0.2)
        
        wave_freq = np.random.uniform(0.1, 0.25)
        wave_amp = np.random.uniform(3, 10)
        
        for i in range(streak_len):
            progress = i / streak_len
            x = pipe_x - i
            
            wave = wave_amp * np.sin(i * wave_freq + si) * progress
            y_center = start_y + int(angle * i + wave)
            
            local_w = int(streak_width * (1 - progress * 0.6))
            
            for dy in range(-local_w, local_w + 1):
                y = y_center + dy
                
                # Random gaps for torn look
                if np.random.random() > 0.92:
                    continue
                
                if 0 <= y < h and 0 <= x < w:
                    d = abs(dy) / (local_w + 0.1)
                    
                    color = get_flame_color(progress, d)
                    
                    a = (1 - d * 0.5) * (1 - progress * 0.5) * intensity
                    a *= (0.7 + 0.3 * np.random.random())
                    a = min(1.0, max(0, a))
                    
                    if a > 0.02:
                        flame[y, x] = np.clip(flame[y, x] + color * a * 0.4, 0, 255)
                        alpha[y, x] = min(1.0, alpha[y, x] + a * 0.35)
    
    # 3. Sparks
    n_sparks = np.random.randint(8, 16)
    for _ in range(n_sparks):
        sx = pipe_x - np.random.randint(5, length//2)
        sy = cy + np.random.randint(-height//3, height//3)
        
        for t in range(np.random.randint(3, 8)):
            px = sx - t * 2
            py = sy + np.random.randint(-1, 2)
            if 0 <= px < w and 0 <= py < h:
                bright = 1 - t / 8
                flame[py, px] = np.clip(flame[py, px] + COLOR_INNER * bright, 0, 255)
                alpha[py, px] = min(1.0, alpha[py, px] + 0.4 * bright)
    
    # Motion blur
    kernel = np.zeros((3, 9))
    kernel[1, :] = 1.0 / 9
    flame = cv2.filter2D(flame, -1, kernel)
    alpha = cv2.filter2D(alpha, -1, kernel)
    
    # Soft blur
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
        tx = x - 50
    else:
        tx = x - fw + 50
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
    
    # Additive blend for glow
    for c in range(3):
        blended = bg[:,:,c] * (1 - a * 0.6) + fg[:,:,c] * a
        blended += fg[:,:,c] * a * 0.25
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
    left_off = (int(-car_w * 0.28), 0)
    right_off = (int(car_w * 0.28), 0)
    
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
        left_len = 130 + int(40 * np.sin(t * 1.1))
        left_h = 55 + int(15 * np.sin(t * 0.8 + 1))
        right_len = 150 + int(50 * np.sin(t * 0.9 + 2))
        right_h = 50 + int(18 * np.sin(t * 1.0 + 1.5))
        
        intensity = 0.85 + 0.25 * np.sin(t * 0.6)
        
        left_flame, left_alpha = create_flame(left_len, left_h, intensity, i * 53)
        right_flame, right_alpha = create_flame(right_len, right_h, intensity, i * 53 + 8000)
        
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
