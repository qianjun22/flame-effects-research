# Gemini Colors (Physically-Based)

**Rating: ⭐⭐⭐**

## Concept

Use Gemini Vision API consultation to determine physically accurate flame colors based on temperature distribution. Avoids HSV→BGR conversion artifacts (blue edge problem).

## Gemini-Provided BGR Values

Based on flame temperature physics:

```python
# BGR format (OpenCV)
colors = {
    'core':       (255, 255, 255),  # White - overexposed
    'inner':      (0, 255, 255),    # Yellow - hottest visible
    'mid':        (0, 165, 255),    # Orange - medium
    'outer':      (0, 69, 255),     # Deep orange - cooler
    'tips':       (0, 0, 200),      # Red - coolest
}
```

## Why This Matters

Early iterations using HSV color space had **blue edge artifacts** when converting back to BGR. Direct BGR specification eliminates this issue.

## Temperature Zones

| Zone | Temperature | Appearance |
|------|-------------|------------|
| Core | >1400°C | White (overexposed) |
| Inner | 1100-1400°C | Bright yellow |
| Mid | 800-1100°C | Orange |
| Outer | 500-800°C | Red-orange |
| Tips | <500°C | Dark red |

## Integration

These color values are now integrated into other methods (cloud, vertical, etc.) for consistent, physically-plausible appearance.

## Output

See `../../outputs/` for methods using these colors.
