# Fireball Round

**Rating: ⭐⭐⭐**

## Concept

Simple radial gradient fireballs - white→yellow→orange→red with heavy Gaussian blur. Clean and artifact-free but potentially too smooth/artificial.

## Key Technique

### Radial Gradient
```python
dist = np.sqrt((x - cx)**2 + (y - cy)**2) / radius
color = interpolate_gradient(dist, [white, yellow, orange, red])
```

### Plate-Based Anchoring
Uses license plate bbox estimation for more stable positioning than car bbox alone:
- `plate_w = car_w * 0.26`
- `plate_y = car_y2 - car_h * 0.2`
- Flames positioned between car edge and plate edge

## Pros
- Very clean, no artifacts
- Natural round appearance
- Looks like actual exhaust backfire

## Cons
- Too smooth/perfect
- Lacks turbulent edges
- May appear artificial in some contexts

## Output

See `../../outputs/03_fireball_round.mp4`
