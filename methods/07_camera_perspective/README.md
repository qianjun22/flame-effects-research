# Camera Perspective Flames

**Rating: ⭐⭐⭐⭐**

## Concept

Flames that appear to **shoot directly toward the viewer** from exhaust pipes. Unlike horizontal flames that blow toward each other, these flames project outward with proper perspective expansion.

## Key Technique

### Perspective Expansion
```python
# Vertical: strongest at exhaust (top), fading toward camera (down)
y_norm = y_coords / height  # 0 at top, 1 at bottom
y_factor = np.power(1 - y_norm, 0.5)

# Horizontal: narrow at exhaust, wide toward viewer (perspective)
spread = 0.3 + 1.5 * y_norm  # Expands as flame approaches camera
x_factor = np.exp(-3 * (x_dist / spread) ** 2)
```

### Multi-Layer Turbulence
```python
turb1 = create_noise_texture(width, height, scale=25, seed=...)  # Large
turb2 = create_noise_texture(width, height, scale=12, seed=...)  # Medium
turb3 = create_noise_texture(width, height, scale=6, seed=...)   # Fine
turbulence = 0.45 * turb1 + 0.35 * turb2 + 0.20 * turb3
```

### Temperature-Based Coloring
- White core (>0.8 intensity)
- Yellow (0.6-0.8)
- Orange (0.2-0.6)
- Red edges (<0.2)

## Improvements Over Previous Versions

1. **No sparks** - Eliminates black dot artifacts
2. **Perspective correct** - Flames expand toward viewer naturally
3. **Independent L/R flames** - Different flickering phases and sizes
4. **Smooth YOLO tracking** - 85/15 bbox smoothing for stability

## Usage

```bash
python code.py
# Outputs: fixed_camera_h264.mp4
```

## Notes

- Designed for rear-view exhaust footage
- Screen blending works well in daylight scenes
- Bloom adds natural glow effect
