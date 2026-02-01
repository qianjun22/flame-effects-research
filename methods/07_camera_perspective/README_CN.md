# 朝向摄像机的火焰透视效果

**评分: ⭐⭐⭐⭐**

## 概念

火焰从排气管**直接朝向观众方向喷射**。与向彼此喷射的水平火焰不同，这些火焰以正确的透视扩展向外投射。

## 核心技术

### 透视扩展
```python
# 垂直方向：排气管处（顶部）最强，向摄像机方向（下方）衰减
y_norm = y_coords / height  # 顶部为0，底部为1
y_factor = np.power(1 - y_norm, 0.5)

# 水平方向：排气管处窄，向观众方向展开（透视效果）
spread = 0.3 + 1.5 * y_norm  # 随着火焰接近摄像机而扩展
x_factor = np.exp(-3 * (x_dist / spread) ** 2)
```

### 多层湍流
```python
turb1 = create_noise_texture(width, height, scale=25, seed=...)  # 大尺度
turb2 = create_noise_texture(width, height, scale=12, seed=...)  # 中尺度
turb3 = create_noise_texture(width, height, scale=6, seed=...)   # 细节
turbulence = 0.45 * turb1 + 0.35 * turb2 + 0.20 * turb3
```

### 基于温度的着色
- 白色核心（强度>0.8）
- 黄色（0.6-0.8）
- 橙色（0.2-0.6）
- 红色边缘（<0.2）

## 相比之前版本的改进

1. **无火花** - 消除黑点伪影
2. **透视正确** - 火焰自然地向观众扩展
3. **左右火焰独立** - 不同的闪烁相位和大小
4. **平滑的YOLO跟踪** - 85/15边界框平滑以保持稳定性

## 使用方法

```bash
python code.py
# 输出: fixed_camera_h264.mp4
```

## 注意事项

- 专为后视排气管镜头设计
- 屏幕混合在日光场景中效果良好
- 泛光效果增加自然光晕
