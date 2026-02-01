# 云状翻滚火焰

**评分: ⭐⭐⭐⭐ (最佳整体效果)**

## 概念

将排气回火视为**湍流云状爆炸**而非定向喷射。真实的排气回火更像是球形爆发而非气流。

## 关键技术

### 多八度湍流
```python
for i, (scale, weight) in enumerate([(60, 0.35), (30, 0.30), (15, 0.20), (8, 0.15)]):
    noise = np.random.rand(height // scale + 2, width // scale + 2)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    # 每个八度的动画扭曲
    dx = int(8 * math.sin(time * (i+1) * 0.4 + i))
    result += weight * noise
```

### 椭圆云形状（非锥形）
```python
ellipse = np.exp(-2 * (dist_x**2 + dist_y**2 * 1.5))
shape = ellipse * turb * (0.5 + 0.5 * turb2)
```

### 基于温度的颜色
| 强度 | 颜色 | 区域 |
|------|------|------|
| >0.8 | 白色 (255,255,255) | 过曝核心 |
| 0.65-0.8 | 白黄色 | 最热可见区 |
| 0.5-0.65 | 亮黄色 | 高温 |
| 0.35-0.5 | 黄橙色 | 中温 |
| 0.2-0.35 | 橙色 | 较冷 |
| 0.08-0.2 | 红橙色 | 边缘 |
| 0-0.08 | 暗红色 | 消散区 |

### 重度辉光
```python
b1 = cv2.GaussianBlur(img, (81, 81), 0)
b2 = cv2.GaussianBlur(img, (41, 41), 0)
result = img + 0.5 * b1 + 0.3 * b2
```

## 为什么有效

1. **不是喷射/光束** - 真实排气回火是爆炸，不是气流
2. **白色核心** - 过曝中心符合真实火焰摄影
3. **湍流边缘** - 多八度噪声创造有机、蓬松边界
4. **叠加混合** - 火焰添加光线，不替换背景

## 参数

- `flame_w, flame_h = 350, 250` - 云尺寸（宽大于高）
- `intensity = 0.75 + 0.25 * sin(t)` - 脉动亮度
- 15%概率随机爆发
- 火花：每帧5-15个粒子，60%生成率

## 输出

见 `../../outputs/01_cloud_billowing.mp4`
