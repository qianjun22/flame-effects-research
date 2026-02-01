# Vertical Flames Toward Camera

**Rating: ⭐⭐⭐**

## Concept

For rear-mounted exhaust viewed from behind, flames should project **toward the camera** (downward in 2D space). This creates perspective-correct flame direction.

## Key Technique

### Vertical Gradient with Spread
```python
# Strongest at top (exhaust), fading downward (toward camera)
y_factor = 1 - (y_coords / height)
y_factor = np.power(y_factor, 0.3)  # Soft falloff

# Wider spread as flame travels toward camera
spread_factor = 1 + 0.8 * (y_coords / height)
x_factor_spread = np.exp(-2 * (x_dist / spread_factor) ** 2)
```

### Screen Blending for Daylight
```python
screen = 1 - (1 - flame) * (1 - background)
blended = background * (1 - alpha) + screen * alpha
```

## Notes

- Original version had spark particles causing black dot artifacts
- This version removes sparks for clean output
- Flame dimensions: 140x200 (taller than wide)

## Output

See `../../outputs/02_vertical_toward_camera.mp4`
