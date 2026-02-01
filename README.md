# Exhaust Flame Effects Research

Research and experimentation on procedural flame effects for video editing AI evaluation.

## Problem Statement

Video editing AI models struggle with flame/fire effects, particularly:
- **Spatial anchoring**: Effects drift away from attachment points over time
- **Temporal consistency**: Frame-to-frame jittering and instability
- **Physical realism**: Flames look artificial or computer-generated

This repo documents various approaches to generating realistic exhaust flame effects, exploring what works and what doesn't.

## Methods Explored

| Method | Description | Quality | Notes |
|--------|-------------|---------|-------|
| [Cloud Billowing](methods/01_cloud_billowing/) | Turbulent cloud-like explosions | ⭐⭐⭐⭐ | Best overall - organic, puffy appearance |
| [Vertical Toward Camera](methods/02_vertical_toward_camera/) | Flames projecting toward viewer | ⭐⭐⭐ | Good perspective, removed spark artifacts |
| [Fireball Round](methods/03_fireball_round/) | Simple radial gradient fireballs | ⭐⭐⭐ | Clean but too smooth/artificial |
| [Irregular Flames](methods/04_irregular_flames/) | Jagged, wind-torn edges | ⭐⭐ | More realistic shape, harder to tune |
| [Gemini Colors](methods/05_gemini_colors/) | Physically-based color temperatures | ⭐⭐⭐ | Accurate colors, integrated into other methods |
| [Exhaust Glow](methods/06_exhaust_glow/) | Glowing pipe interior with streaks | ⭐⭐⭐⭐ | Realistic - simulates looking into hot pipe |
| [Camera Perspective](methods/07_camera_perspective/) | Flames shooting toward viewer | ⭐⭐⭐⭐ | Perspective-correct expansion, no sparks |

## Key Techniques

### 1. Multi-Octave Turbulence
Layer 4 noise octaves at different scales (60px, 30px, 15px, 8px) with animated phase shifts for organic, billowing motion.

### 2. Temperature-Based Color Gradient
Map intensity to physically accurate flame colors:
- White core (>0.8) - overexposed hottest region
- Yellow (0.5-0.8) - hot visible flame
- Orange (0.2-0.5) - medium temperature
- Red (0-0.2) - cooler edges

### 3. YOLO Car Tracking
YOLOv8 object detection with 85/15 temporal smoothing to anchor flames to vehicle position.

### 4. Additive/Screen Blending
Fire adds light rather than replacing pixels - use additive or screen blend modes.

### 5. Heavy Bloom
Double Gaussian blur (81px + 41px kernels) creates characteristic fire glow.

## Requirements

```bash
pip install opencv-python numpy ultralytics
```

## Usage

Each method folder contains:
- `code.py` - The implementation
- `README.md` - Method explanation
- Sample frames in `outputs/`

```bash
cd methods/01_cloud_billowing
python code.py
```

## Outputs

- `outputs/` - Generated video files
- `comparisons/` - Side-by-side comparison videos

## Insights for Model Training

1. **Spatial anchoring supervision**: Models need explicit anchor point conditioning
2. **Two-stage pipeline**: Generate effects → warp to tracked positions
3. **Physics constraints**: Flame effects may need specialized modules beyond image synthesis
4. **Temporal coherence loss**: Penalize frame-to-frame discontinuities

## Limitations of 2D Procedural Generation

Gemini Vision consistently requested features requiring 3D VFX:
- Fluid simulation (Houdini, EmberGen)
- Heat distortion (3D displacement)
- Volumetric rendering
- Light interaction with surfaces

**Conclusion**: Production-quality flames likely need specialized tools or learned 3D representations, not just 2D compositing.

## License

MIT - Free for research and commercial use.
