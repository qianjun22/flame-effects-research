# Irregular Flames

**Rating: ⭐⭐**

## Concept

Jagged, wind-torn edges for more realistic racing car backfire appearance. Not round blobs - irregular, streaming flames.

## Requirements from User Feedback

1. Irregular edges (not round/blob-like)
2. Flowing/streaming appearance (wind-blown)
3. 3D depth (bright core, darker edges)
4. Left and right flames different sizes
5. Dynamic frame-to-frame size changes
6. Fire must emerge FROM exhaust pipe
7. Elongated horizontal jets

## Key Techniques

### Asymmetric Sizing
```python
# Different sinusoidal cycles for L/R independence
size_l = base_size * (0.8 + 0.4 * sin(t * 0.35))
size_r = base_size * (0.8 + 0.4 * sin(t * 0.45 + 1.3))
```

### Wind-Torn Edges
Multiple turbulence layers with different frequencies create irregular boundaries.

## Challenges

- Harder to tune than smooth approaches
- Risk of artifacts at extreme settings
- Balance between realism and stability

## Output

Experimental - see code for current implementation.
