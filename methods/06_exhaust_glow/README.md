# Exhaust Glow

**Rating: ⭐⭐⭐⭐ (Highly Realistic)**

## Concept

Simulates looking into a **glowing hot exhaust pipe** with flames shooting outward. The key insight is that real exhaust flames have a bright radial core at the pipe opening, not just flames floating in space.

## Key Techniques

### 1. Glowing Pipe Interior
Creates a bright radial gradient simulating the hot interior of the exhaust pipe:

```python
# Inside pipe - very bright white/yellow core
if dist <= inner_r:
    if t < 0.3:
        color = WHITE_CORE  # Hottest center
    elif t < 0.6:
        color = WHITE_CORE * 0.7 + YELLOW * 0.3
    else:
        color = YELLOW * 0.8 + ORANGE * 0.2
```

### 2. Flame Streaks Shooting Out
Multiple turbulent streaks emanate from the pipe opening:

```python
n_streaks = np.random.randint(8, 14)  # Variable number
start_r = np.random.uniform(0, inner_r * 0.7)  # Start from within pipe
streak_len = int(length * np.random.uniform(0.4, 0.9))  # Variable length
```

### 3. Horizontal Motion Blur
Simulates the fast-moving nature of exhaust gases:

```python
kernel = np.zeros((5, 21))
kernel[2, :] = 1.0 / 21  # Horizontal blur
flame = cv2.filter2D(flame, -1, kernel)
```

### 4. Screen Blending
Proper light-additive compositing:

```python
blended = bg + fg * alpha - (bg * fg * alpha) / 255
```

## Color Palette (BGR)

| Zone | Color | BGR Value |
|------|-------|-----------|
| Core | White | (255, 255, 255) |
| Inner | Yellow | (0, 255, 255) |
| Mid | Orange | (0, 165, 255) |
| Outer | Deep Orange | (0, 100, 255) |
| Tips | Red | (0, 50, 200) |

## Parameters

- `left_len`: 120 ± 35px (animated)
- `right_len`: 140 ± 45px (animated)
- `intensity`: 0.85 ± 0.25 (pulsing)
- Streaks per flame: 8-14

## Why It Works

1. **Radial core** - Looks like you're seeing into the hot pipe
2. **Motion blur** - Conveys speed and heat distortion
3. **Proper colors** - Temperature-based gradient
4. **Asymmetric L/R** - Natural variation between exhausts

## Output

See `../../outputs/06_exhaust_glow.mp4`
