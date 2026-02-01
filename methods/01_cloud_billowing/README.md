# Cloud Billowing Flames

**Rating: ⭐⭐⭐⭐ (Best Overall)**

## Concept

Treat exhaust backfire as a **turbulent cloud explosion** rather than a directed jet. Real exhaust backfire is more like a spherical burst than a stream.

## Key Techniques

### Multi-Octave Turbulence
```python
for i, (scale, weight) in enumerate([(60, 0.35), (30, 0.30), (15, 0.20), (8, 0.15)]):
    noise = np.random.rand(height // scale + 2, width // scale + 2)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    # Animated distortion per octave
    dx = int(8 * math.sin(time * (i+1) * 0.4 + i))
    result += weight * noise
```

### Elliptical Cloud Shape (Not Cone)
```python
ellipse = np.exp(-2 * (dist_x**2 + dist_y**2 * 1.5))
shape = ellipse * turb * (0.5 + 0.5 * turb2)
```

### Temperature-Based Colors
| Intensity | Color | Zone |
|-----------|-------|------|
| >0.8 | White (255,255,255) | Overexposed core |
| 0.65-0.8 | White-yellow | Hottest visible |
| 0.5-0.65 | Bright yellow | Hot |
| 0.35-0.5 | Yellow-orange | Medium |
| 0.2-0.35 | Orange | Cooler |
| 0.08-0.2 | Red-orange | Edge |
| 0-0.08 | Dark red | Dissipating |

### Heavy Bloom
```python
b1 = cv2.GaussianBlur(img, (81, 81), 0)
b2 = cv2.GaussianBlur(img, (41, 41), 0)
result = img + 0.5 * b1 + 0.3 * b2
```

## Why It Works

1. **Not a jet/beam** - Real exhaust backfire is an explosion, not a stream
2. **White core** - Overexposed center matches real fire photography
3. **Turbulent edges** - Multi-octave noise creates organic, puffy boundaries
4. **Additive blend** - Fire adds light, doesn't replace background

## Parameters

- `flame_w, flame_h = 350, 250` - Cloud dimensions (wider than tall)
- `intensity = 0.75 + 0.25 * sin(t)` - Pulsing brightness
- 15% chance of random burst
- Sparks: 5-15 particles per frame, 60% spawn rate

## Output

See `../../outputs/01_cloud_billowing.mp4`
