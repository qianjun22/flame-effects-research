#!/usr/bin/env python3
"""
Irregular, jagged exhaust flames - not smooth/round.
Uses noise and random patterns for realistic turbulent look.
"""
import cv2
import numpy as np
from ultralytics import YOLO

RAW = "/tmp/video_editing_problem/Video Editing问题数据样例/1105_raw.mp4"
OUT = "/tmp/flame_fix/fixed_irregular"

def perlin_noise_1d(x, freq=1.0, seed=0):
    """Simple 1D noise function."""
    np.random.seed(seed)
    return np.sin(x * freq + np.random.random() * 10) * np.cos(x * freq * 0.7 + np.random.random() * 5)

def create_irregular_flame(base_length=100, base_height=45, intensity=1.0, seed=0):
    """Create flame with irregular, jagged edges."""
    np.random.seed(seed)
    
    pad = 50
    length = base_length + np.random.randint(-15, 15)
    height = base_height + np.random.randint(-8, 8)
    
    w = length + 2 * pad
    h = height + 2 * pad
    
    flame = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    
    cy = h // 2
    
    # Create multiple irregular "tongues" of flame
    n_tongues = np.random.randint(4, 8)
    tongue_angles = np.random.uniform(-0.4, 0.4, n_tongues)
    tongue_lengths = np.random.uniform(0.6, 1.2, n_tongues)
    tongue_widths = np.random.uniform(0.5, 1.0, n_tongues)
    
    for ti in range(n_tongues):
        angle = tongue_angles[ti]
        t_len = int(length * tongue_lengths[ti])
        t_width = height * tongue_widths[ti]
        
        # Draw this tongue
        for x in range(pad, min(pad + t_len, w - pad)):
            t = (x - pad) / t_len
            
            # Irregular height using noise
            noise1 = np.sin(t * 12 + seed * ti) * 0.3
            noise2 = np.sin(t * 25 + seed * ti * 2) * 0.15
            noise3 = np.random.random() * 0.1
            
            local_h = int(t_width * (1 - t * 0.5) * (1 + noise1 + noise2) / 2)
            
            # Offset based on angle
            y_offset = int(angle * (x - pad))
            
            for dy in range(-local_h, local_h + 1):
                y = cy + dy + y_offset
                
                # Add jagged edges
                edge_noise = np.random.random()
                if edge_noise > 0.85 and abs(dy) > local_h * 0.6:
                    continue  # Skip some edge pixels for jaggedness
                
                if 0 <= y < h:
                    d = abs(dy) / (local_h + 0.1)
                    
                    # Irregular brightness
                    bright_noise = 0.7 + 0.3 * np.random.random()
                    
                    # BGR colors
                    if t < 0.2:  # Core
                        if d < 0.3 and np.random.random() > 0.3:
                            color = np.array([200, 255, 255]) * bright_noise  # White-yellow
                        elif d < 0.5:
                            color = np.array([50, 220, 255]) * bright_noise   # Yellow
                        else:
                            color = np.array([30, 170, 255]) * bright_noise   # Orange
                    elif t < 0.5:
                        if d < 0.4:
                            color = np.array([40, 200, 255]) * bright_noise   # Yellow
                        else:
                            color = np.array([20, 130, 255]) * bright_noise   # Orange
                    elif t < 0.8:
                        if d < 0.5:
                            color = np.array([25, 150, 255]) * bright_noise   # Orange
                        else:
                            color = np.array([10, 90, 255]) * bright_noise    # Red-orange
                    else:
                        if d < 0.6:
                            color = np.array([15, 100, 255]) * bright_noise   # Orange
                        else:
                            color = np.array([0, 50, 220]) * bright_noise     # Red
                    
                    # Alpha with noise
                    a = (1 - d * 0.8) * (1 - t * 0.4) * intensity
                    a = a * (0.8 + 0.2 * np.random.random())
                    a = min(1.0, max(0, a))
                    
                    # Accumulate (brighter overwrites)
                    if a > alpha[y, x]:
                        flame[y, x] = np.clip(color, 0, 255)
                        alpha[y, x] = a
    
    # Add hot spots/sparks
    n_sparks = np.random.randint(5, 15)
    for _ in range(n_sparks):
        sx = np.random.randint(pad, pad + length // 2)
        sy = np.random.randint(cy - height//3, cy + height//3)
        spark_r = np.random.randint(2, 5)
        
        for dx in range(-spark_r, spark_r + 1):
            for dy in range(-spark_r, spark_r + 1):
                if dx*dx + dy*dy <= spark_r*spark_r:
                    px, py = sx + dx, sy + dy
                    if 0 <= px < w and 0 <= py < h:
                        flame[py, px] = [220, 255, 255]  # Bright white-yellow
                        alpha[py, px] = min(1.0, alpha[py, px] + 0.5)
    
    # Light blur to blend (but keep some sharpness)
    flame = cv2.GaussianBlur(flame, (9, 9), 0)
    alpha = cv2.GaussianBlur(alpha, (11, 11), 0)
    
    return flame.astype(np.uint8), alpha

def paste_flame(frame, flame_bgr, flame_alpha, x, y):
    """Paste flame at exhaust position."""
    fh, fw = flame_bgr.shape[:2]
    h, w = frame.shape[:2]
    
    tx = x - fw + 50
    ty = y - fh // 2
    
    src_x1 = max(0, -tx)
    src_y1 = max(0, -ty)
    dst_x1 = max(0, tx)
    dst_y1 = max(0, ty)
    
    src_x2 = min(fw, w - tx)
    src_y2 = min(fh, h - ty)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return frame
    
    fg = flame_bgr[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
    a = flame_alpha[src_y1:src_y2, src_x1:src_x2]
    
    result = frame.copy()
    bg = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    
    for c in range(3):
        blended = bg[:,:,c] * (1 - a) + fg[:,:,c] * a
        blended = np.clip(blended + fg[:,:,c] * a * 0.2, 0, 255)  # Slight glow
        result[dst_y1:dst_y2, dst_x1:dst_x2, c] = blended.astype(np.uint8)
    
    return result

def main():
    print("Loading video...")
    cap = cv2.VideoCapture(RAW)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {n_frames} frames, {w}x{h}")
    
    print("Loading YOLO...")
    model = YOLO('yolov8n.pt')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT + '_temp.mp4', fourcc, fps, (w, h))
    
    # Reference car
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    results = model(frame0, classes=[2], verbose=False)
    
    ref_car = None
    for r in results:
        for box in r.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            if (bx2 - bx1) > 150:
                ref_car = (bx1, by1, bx2, by2)
                break
    
    if ref_car is None:
        ref_car = (888, 407, 1159, 595)
    
    car_w = ref_car[2] - ref_car[0]
    left_offset = (int(-car_w * 0.22), 5)
    right_offset = (int(car_w * 0.22), 5)
    
    print(f"Exhaust offsets: L={left_offset}, R={right_offset}")
    
    print("Processing frames...")
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
        
        left_x = car_cx + left_offset[0]
        left_y = car_bottom + left_offset[1]
        right_x = car_cx + right_offset[0]
        right_y = car_bottom + right_offset[1]
        
        # Create irregular flames
        intensity = 0.85 + 0.3 * np.sin(i * 0.3)
        left_flame, left_alpha = create_irregular_flame(
            base_length=110, base_height=50, intensity=intensity, seed=i * 7
        )
        right_flame, right_alpha = create_irregular_flame(
            base_length=110, base_height=50, intensity=intensity, seed=i * 7 + 100
        )
        
        output = frame.copy()
        output = paste_flame(output, left_flame, left_alpha, left_x, left_y)
        output = paste_flame(output, right_flame, right_alpha, right_x, right_y)
        
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
    
    print(f"\nDone: {OUT}_h264.mp4")

if __name__ == "__main__":
    main()
