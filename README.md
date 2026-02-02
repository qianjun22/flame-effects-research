# Exhaust Flame Effects Research

Research and experimentation on procedural flame effects for video editing AI evaluation.

<p align="center">
  <img src="https://img.shields.io/badge/Methods-7-blue" alt="Methods">
  <img src="https://img.shields.io/badge/Videos-62-green" alt="Videos">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

## 📥 Downloads

**[⬇️ Download All Videos (370MB)](https://github.com/qianjun22/flame-effects-research/releases/tag/v1.0.0-videos)** - 62 flame effect videos including comparisons and original footage.

## Problem Statement

Video editing AI models struggle with flame/fire effects, particularly:
- **Spatial anchoring**: Effects drift away from attachment points over time
- **Temporal consistency**: Frame-to-frame jittering and instability
- **Physical realism**: Flames look artificial or computer-generated

This repo documents various approaches to generating realistic exhaust flame effects, exploring what works and what doesn't.

## Methods Explored

| # | Method | Description | Quality | Notes |
|---|--------|-------------|---------|-------|
| 01 | [Cloud Billowing](methods/01_cloud_billowing/) | Turbulent cloud-like explosions | ⭐⭐⭐⭐ | Best overall - organic, puffy appearance |
| 02 | [Vertical Toward Camera](methods/02_vertical_toward_camera/) | Flames projecting toward viewer | ⭐⭐⭐ | Good perspective, spark artifacts removed |
| 03 | [Fireball Round](methods/03_fireball_round/) | Simple radial gradient fireballs | ⭐⭐⭐ | Clean but too smooth/artificial |
| 04 | [Irregular Flames](methods/04_irregular_flames/) | Jagged, wind-torn edges | ⭐⭐ | More realistic shape, harder to tune |
| 05 | [Gemini Colors](methods/05_gemini_colors/) | Physically-based color temperatures | ⭐⭐⭐ | Accurate colors, integrated into other methods |
| 06 | [Exhaust Glow](methods/06_exhaust_glow/) | Glowing pipe interior with streaks | ⭐⭐⭐⭐ | Realistic - simulates looking into hot pipe |
| 07 | [Camera Perspective](methods/07_camera_perspective/) | Flames shooting toward viewer | ⭐⭐⭐⭐ | Perspective-correct expansion, no sparks |

## Key Techniques

### 1. Multi-Octave Turbulence
```python
# Layer 4 noise octaves at different scales
turb1 = create_noise(width, height, scale=60)  # Large structures
turb2 = create_noise(width, height, scale=30)  # Medium detail
turb3 = create_noise(width, height, scale=15)  # Fine detail
turb4 = create_noise(width, height, scale=8)   # Micro detail
turbulence = 0.4*turb1 + 0.3*turb2 + 0.2*turb3 + 0.1*turb4
```

### 2. Temperature-Based Color Gradient
Map intensity to physically accurate flame colors:
| Temperature | Intensity | Color (BGR) |
|-------------|-----------|-------------|
| Hottest | >0.8 | White (255, 255, 255) |
| Hot | 0.5-0.8 | Yellow (0, 255, 255) |
| Medium | 0.2-0.5 | Orange (0, 165, 255) |
| Cool | <0.2 | Red (0, 0, 200) |

### 3. YOLO Car Tracking
```python
# YOLOv8 detection with temporal smoothing
model = YOLO('yolov8n.pt')
results = model(frame, classes=[2])  # class 2 = car
# 85/15 smoothing for stability
bbox = 0.85 * prev_bbox + 0.15 * current_bbox
```

### 4. Screen Blending
Fire adds light rather than replacing pixels:
```python
# Screen blend mode
blended = 1 - (1 - foreground) * (1 - background)
result = background * (1 - alpha) + blended * alpha
```

### 5. Heavy Bloom
Double Gaussian blur creates characteristic fire glow:
```python
bloom1 = cv2.GaussianBlur(flame, (81, 81), 0)
bloom2 = cv2.GaussianBlur(flame, (41, 41), 0)
result = cv2.addWeighted(flame, 1.0, bloom1, 0.5, 0)
result = cv2.addWeighted(result, 1.0, bloom2, 0.3, 0)
```

## Requirements

```bash
pip install opencv-python numpy ultralytics
```

## Usage

Each method folder contains:
- `code.py` - The implementation
- `README.md` - English explanation
- `README_CN.md` - Chinese explanation

```bash
# Run any method
cd methods/07_camera_perspective
python code.py
# Output: fixed_camera_h264.mp4
```

## Project Structure

```
flame-effects-research/
├── methods/
│   ├── 01_cloud_billowing/
│   ├── 02_vertical_toward_camera/
│   ├── 03_fireball_round/
│   ├── 04_irregular_flames/
│   ├── 05_gemini_colors/
│   ├── 06_exhaust_glow/
│   └── 07_camera_perspective/
├── inputs/                  # Original test videos
├── outputs/                 # Generated results
├── comparisons/             # Side-by-side comparisons
├── README.md
└── README_CN.md
```

## Insights for Model Training

1. **Spatial anchoring supervision**: Models need explicit anchor point conditioning
2. **Two-stage pipeline**: Generate effects → warp to tracked positions
3. **Physics constraints**: Flame effects may need specialized modules beyond image synthesis
4. **Temporal coherence loss**: Penalize frame-to-frame discontinuities
5. **Perspective awareness**: Effects should scale/expand based on distance from camera

## Limitations of 2D Procedural Generation

Gemini Vision consistently requested features requiring 3D VFX:
- Fluid simulation (Houdini, EmberGen)
- Heat distortion (3D displacement)
- Volumetric rendering
- Light interaction with surfaces

**Conclusion**: Production-quality flames likely need specialized tools or learned 3D representations, not just 2D compositing.

## Video Collection

The [v1.0.0-videos release](https://github.com/qianjun22/flame-effects-research/releases/tag/v1.0.0-videos) contains:

| Category | Count | Description |
|----------|-------|-------------|
| Original | 1 | Raw input video (1105_raw.mp4) |
| H264 Outputs | 54 | Final encoded flame effect videos |
| Comparisons | 6 | Side-by-side comparison videos |
| **Total** | **62** | **370MB zipped** |

## License

MIT - Free for research and commercial use.

## Contributing

Contributions welcome! Feel free to:
- Add new flame generation methods
- Improve existing implementations
- Share evaluation results
- Suggest training data approaches
