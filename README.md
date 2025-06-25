# 滑块模板匹配程序

这是一个基于OpenCV的滑块验证码模板匹配程序，可以识别图像中的字母序列并模拟滑动操作。

## 功能特点

- 🔍 **模板匹配**: 使用OpenCV进行高精度模板匹配
- 📝 **字母识别**: 支持识别A、D、S、W四个字母
- 🎯 **序列检测**: 自动检测连续的字母序列
- 📏 **滑动模拟**: 模拟滑块滑动指定距离（默认50px）
- 🎨 **结果可视化**: 生成带有标注的结果图像

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备模板图片

确保 `template_images` 文件夹中包含以下模板文件：
- `a.png` - 字母A的模板
- `d.png` - 字母D的模板  
- `s.png` - 字母S的模板
- `w.png` - 字母W的模板

### 2. 运行程序

```bash
python main.py
```

程序会提示输入目标图像路径，输入后会自动进行匹配和分析。

### 3. 编程接口使用

```python
from main import SliderTemplateMatching

# 创建匹配器
matcher = SliderTemplateMatching()

# 执行匹配
results = matcher.find_slider_sequence("your_image.png", slide_distance=50)

# 查看结果
if results["sequence_found"]:
    print("检测到序列:", results["detected_sequence"])
    print("滑块位置:", results["slider_position"])
else:
    print("未检测到有效序列")

# 生成可视化结果
matcher.visualize_results("your_image.png", results, "output.png")
```

## 输出结果

程序会输出以下信息：

1. **匹配结果**: 每个模板在图像中的匹配位置和置信度
2. **字母序列**: 检测到的连续字母序列
3. **滑块操作**: 起始位置、目标位置和滑动距离
4. **可视化图像**: 保存为 `result.png`，包含：
   - 彩色矩形框标注匹配的字母
   - 箭头显示滑动方向和距离
   - 置信度标签

## 参数说明

- `slide_distance`: 滑动距离（像素），默认50
- `threshold`: 匹配阈值，默认0.7，范围0-1
- `template_dir`: 模板图片目录，默认"template_images"

## 注意事项

1. 确保目标图像清晰，字母可见
2. 模板图片应该是单个字母的清晰截图
3. 程序会自动调整匹配阈值以获得最佳效果
4. 支持多个相同字母的匹配

## 文件结构

```
black-fish/
├── main.py              # 主程序文件
├── requirements.txt     # 依赖包列表
├── README.md           # 说明文档
└── template_images/    # 模板图片目录
    ├── a.png
    ├── d.png
    ├── s.png
    └── w.png
```