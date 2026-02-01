# 排气管发光效果

**评分: ⭐⭐⭐⭐ (高度真实)**

## 概念

模拟看入**发光灼热排气管**内部，火焰向外喷射。关键洞察是真实的排气火焰在管口有明亮的径向核心，而不仅仅是漂浮在空中的火焰。

## 关键技术

### 1. 发光的管道内部
创建明亮的径向渐变，模拟排气管内部的高温：

```python
# 管内 - 非常明亮的白/黄核心
if dist <= inner_r:
    if t < 0.3:
        color = WHITE_CORE  # 最热的中心
    elif t < 0.6:
        color = WHITE_CORE * 0.7 + YELLOW * 0.3
    else:
        color = YELLOW * 0.8 + ORANGE * 0.2
```

### 2. 向外喷射的火焰条纹
多条湍流条纹从管口发出：

```python
n_streaks = np.random.randint(8, 14)  # 可变数量
start_r = np.random.uniform(0, inner_r * 0.7)  # 从管内开始
streak_len = int(length * np.random.uniform(0.4, 0.9))  # 可变长度
```

### 3. 水平运动模糊
模拟排气气体的快速运动：

```python
kernel = np.zeros((5, 21))
kernel[2, :] = 1.0 / 21  # 水平模糊
flame = cv2.filter2D(flame, -1, kernel)
```

### 4. 屏幕混合
正确的光线叠加合成：

```python
blended = bg + fg * alpha - (bg * fg * alpha) / 255
```

## 调色板 (BGR)

| 区域 | 颜色 | BGR值 |
|------|------|-------|
| 核心 | 白色 | (255, 255, 255) |
| 内层 | 黄色 | (0, 255, 255) |
| 中层 | 橙色 | (0, 165, 255) |
| 外层 | 深橙色 | (0, 100, 255) |
| 尖端 | 红色 | (0, 50, 200) |

## 参数

- `left_len`: 120 ± 35px (动画)
- `right_len`: 140 ± 45px (动画)
- `intensity`: 0.85 ± 0.25 (脉动)
- 每个火焰的条纹数: 8-14

## 为什么有效

1. **径向核心** - 看起来像是看入热管道内部
2. **运动模糊** - 传达速度和热扭曲感
3. **正确颜色** - 基于温度的渐变
4. **左右不对称** - 两个排气口之间的自然变化

## 输出

见 `../../outputs/06_exhaust_glow.mp4`
