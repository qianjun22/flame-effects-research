# 圆形火球

**评分: ⭐⭐⭐**

## 概念

简单径向渐变火球 - 白→黄→橙→红，配合重度高斯模糊。干净无伪影，但可能过于平滑/人工。

## 关键技术

### 径向渐变
```python
dist = np.sqrt((x - cx)**2 + (y - cy)**2) / radius
color = interpolate_gradient(dist, [white, yellow, orange, red])
```

### 基于车牌的锚定
使用车牌边界框估计，比仅用车辆边界框更稳定：
- `plate_w = car_w * 0.26`
- `plate_y = car_y2 - car_h * 0.2`
- 火焰定位在车辆边缘和车牌边缘之间

## 优点
- 非常干净，无伪影
- 自然的圆形外观
- 看起来像真实排气回火

## 缺点
- 过于平滑/完美
- 缺乏湍流边缘
- 某些情况下可能显得人工

## 输出

见 `../../outputs/03_fireball_round.mp4`
