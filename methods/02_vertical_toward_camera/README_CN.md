# 向镜头垂直火焰

**评分: ⭐⭐⭐**

## 概念

对于从后方观看的后置排气，火焰应**向镜头投射**（在2D空间中向下）。这创造了透视正确的火焰方向。

## 关键技术

### 带扩散的垂直渐变
```python
# 顶部（排气口）最强，向下（向镜头）衰减
y_factor = 1 - (y_coords / height)
y_factor = np.power(y_factor, 0.3)  # 柔和衰减

# 火焰向镜头移动时扩散更宽
spread_factor = 1 + 0.8 * (y_coords / height)
x_factor_spread = np.exp(-2 * (x_dist / spread_factor) ** 2)
```

### 日光下的屏幕混合
```python
screen = 1 - (1 - flame) * (1 - background)
blended = background * (1 - alpha) + screen * alpha
```

## 备注

- 原版本有火花粒子导致黑点伪影
- 此版本移除火花以获得干净输出
- 火焰尺寸：140x200（高大于宽）

## 输出

见 `../../outputs/02_vertical_toward_camera.mp4`
