# 排气火焰效果研究

视频编辑AI评估的程序化火焰效果研究与实验。

<p align="center">
  <img src="https://img.shields.io/badge/方法-7-blue" alt="Methods">
  <img src="https://img.shields.io/badge/视频-62-green" alt="Videos">
  <img src="https://img.shields.io/badge/许可证-MIT-yellow" alt="License">
</p>

## 📥 下载

**[⬇️ 下载所有视频 (370MB)](https://github.com/qianjun22/flame-effects-research/releases/tag/v1.0.0-videos)** - 62个火焰效果视频，包括对比视频和原始素材。

## 问题陈述

视频编辑AI模型在处理火焰/火效果时存在困难，主要问题包括：
- **空间锚定**：效果随时间从附着点漂移
- **时间一致性**：帧间抖动和不稳定
- **物理真实感**：火焰看起来不自然或像计算机生成

本仓库记录了生成逼真排气火焰效果的各种方法，探索什么有效什么无效。

## 探索的方法

| # | 方法 | 描述 | 质量 | 备注 |
|---|------|------|------|------|
| 01 | [云状翻滚火焰](methods/01_cloud_billowing/) | 湍流云状爆炸 | ⭐⭐⭐⭐ | 最佳整体效果 - 有机、蓬松外观 |
| 02 | [向镜头垂直火焰](methods/02_vertical_toward_camera/) | 火焰向观众投射 | ⭐⭐⭐ | 透视正确，已移除火花伪影 |
| 03 | [圆形火球](methods/03_fireball_round/) | 简单径向渐变火球 | ⭐⭐⭐ | 干净但过于平滑/人工 |
| 04 | [不规则火焰](methods/04_irregular_flames/) | 锯齿状、风吹边缘 | ⭐⭐ | 形状更真实，但难以调节 |
| 05 | [Gemini颜色](methods/05_gemini_colors/) | 基于物理的色温 | ⭐⭐⭐ | 颜色准确，已整合到其他方法 |
| 06 | [排气管发光](methods/06_exhaust_glow/) | 发光管道内部配合火焰条纹 | ⭐⭐⭐⭐ | 逼真 - 模拟看向高温管道内部 |
| 07 | [摄像机透视火焰](methods/07_camera_perspective/) | 火焰向观众方向喷射 | ⭐⭐⭐⭐ | 透视正确的扩展效果，无火花 |

## 关键技术

### 1. 多八度湍流
```python
# 在不同尺度叠加4个噪声八度
turb1 = create_noise(width, height, scale=60)  # 大结构
turb2 = create_noise(width, height, scale=30)  # 中等细节
turb3 = create_noise(width, height, scale=15)  # 精细细节
turb4 = create_noise(width, height, scale=8)   # 微细节
turbulence = 0.4*turb1 + 0.3*turb2 + 0.2*turb3 + 0.1*turb4
```

### 2. 基于温度的颜色渐变
将强度映射到物理准确的火焰颜色：
| 温度 | 强度 | 颜色 (BGR) |
|------|------|------------|
| 最热 | >0.8 | 白色 (255, 255, 255) |
| 高温 | 0.5-0.8 | 黄色 (0, 255, 255) |
| 中温 | 0.2-0.5 | 橙色 (0, 165, 255) |
| 低温 | <0.2 | 红色 (0, 0, 200) |

### 3. YOLO车辆跟踪
```python
# 使用YOLOv8检测，配合时间平滑
model = YOLO('yolov8n.pt')
results = model(frame, classes=[2])  # class 2 = 汽车
# 85/15平滑以保持稳定
bbox = 0.85 * prev_bbox + 0.15 * current_bbox
```

### 4. 屏幕混合
火焰添加光线而非替换像素：
```python
# 屏幕混合模式
blended = 1 - (1 - foreground) * (1 - background)
result = background * (1 - alpha) + blended * alpha
```

### 5. 重度辉光
双高斯模糊创造特征性火焰光晕：
```python
bloom1 = cv2.GaussianBlur(flame, (81, 81), 0)
bloom2 = cv2.GaussianBlur(flame, (41, 41), 0)
result = cv2.addWeighted(flame, 1.0, bloom1, 0.5, 0)
result = cv2.addWeighted(result, 1.0, bloom2, 0.3, 0)
```

## 环境要求

```bash
pip install opencv-python numpy ultralytics
```

## 使用方法

每个方法文件夹包含：
- `code.py` - 实现代码
- `README.md` - 英文说明
- `README_CN.md` - 中文说明

```bash
# 运行任意方法
cd methods/07_camera_perspective
python code.py
# 输出: fixed_camera_h264.mp4
```

## 项目结构

```
flame-effects-research/
├── methods/
│   ├── 01_cloud_billowing/
│   ├── 02_vertical_toward_camera/
│   ├── 03_fireball_round/
│   ├── 04_irregular_flames/
│   ├── 05_gemini_colors/
│   ├── 06_exhaust_glow/
│   └── 07_camera_perspective/
├── inputs/                  # 原始测试视频
├── outputs/                 # 生成结果
├── comparisons/             # 并排对比
├── README.md
└── README_CN.md
```

## 模型训练的洞察

1. **空间锚定监督**：模型需要显式的锚点条件
2. **两阶段流水线**：生成效果 → 变形到跟踪位置
3. **物理约束**：火焰效果可能需要超越图像合成的专门模块
4. **时间一致性损失**：惩罚帧间不连续性
5. **透视感知**：效果应根据与摄像机的距离进行缩放/扩展

## 2D程序化生成的局限性

Gemini Vision一致请求需要3D VFX的功能：
- 流体模拟（Houdini、EmberGen）
- 热扭曲（3D位移）
- 体积渲染
- 光线与表面交互

**结论**：生产级火焰可能需要专门工具或学习的3D表示，而非仅仅2D合成。

## 视频集合

[v1.0.0-videos发布](https://github.com/qianjun22/flame-effects-research/releases/tag/v1.0.0-videos)包含：

| 类别 | 数量 | 描述 |
|------|------|------|
| 原始视频 | 1 | 原始输入视频 (1105_raw.mp4) |
| H264输出 | 54 | 最终编码的火焰效果视频 |
| 对比视频 | 6 | 并排对比视频 |
| **总计** | **62** | **压缩后370MB** |

## 许可证

MIT - 可自由用于研究和商业用途。

## 贡献

欢迎贡献！您可以：
- 添加新的火焰生成方法
- 改进现有实现
- 分享评估结果
- 建议训练数据方法
