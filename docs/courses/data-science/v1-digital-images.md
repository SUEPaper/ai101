---
course: data-science
title: 数字图像：从像素到数组合同
description: 用 OpenCV 与 NumPy 建立数字图像的形状、类型、颜色、坐标、ROI、文件格式和批次接口。
outline: [2, 3]
---

<DsLessonHero
  number="V1"
  title="数字图像：从像素到数组合同"
  eyebrow="计算机看到的不是照片，而是一组带坐标、通道、类型和取值范围的数"
  duration="建议 120–150 分钟"
  :outcomes="[
    '用 shape、dtype、取值范围和颜色顺序描述一幅图像',
    '正确区分 NumPy 的行列索引与图像的 x、y 坐标',
    '使用 OpenCV 读取、转换、编码和检查图像',
    '把 HWC 图像转换成深度学习常用的 NCHW 浮点批次'
  ]"
/>

## 本讲要解决的问题

同一批产线照片在文件浏览器里看起来正常，送入分析程序后却出现颜色颠倒、亮度异常或尺寸不一致。算法还没有开始，数据合同已经被破坏。本讲先把图像当作一种有空间结构的数组，明确每个轴、每个数和每次转换代表什么。

本模块统一使用 Python。OpenCV 的 Python 包导入名是 `cv2`，课程采用常见别名：

```python
import cv2 as cv
import numpy as np
import pandas as pd

print("OpenCV：", cv.__version__)
```

## 1. 数字图像是采样和量化后的数组

真实场景中的光是连续信号。相机把空间划成有限网格，这是**采样**；再把每个位置的亮度或颜色映射到有限数值，这是**量化**。因此，一幅数字图像至少包含四类合同：

| 合同 | 典型问题 | 错误后果 |
|---|---|---|
| 空间尺寸 | 高、宽分别是多少 | 模型输入形状不一致 |
| 通道语义 | 灰度、BGR、RGB 还是带 Alpha | 颜色颠倒或通道数错误 |
| 数值类型 | `uint8`、`uint16`、`float32` | 溢出、精度丢失、归一化重复 |
| 取值范围 | `0–255`、`0–65535`、`0–1` | 对比度、阈值和模型输入全部改变 |

下面构造一幅 `6 × 8` 的原创三通道教学图。OpenCV 的常用彩色数组按 BGR 排列：

```python
image_bgr = np.zeros((6, 8, 3), dtype=np.uint8)
image_bgr[:] = [28, 46, 62]
image_bgr[1:5, 2:6] = [42, 152, 226]
image_bgr[2:4, 3:5] = [190, 92, 38]

assert image_bgr.shape == (6, 8, 3)
assert image_bgr.dtype == np.uint8
assert 0 <= image_bgr.min() <= image_bgr.max() <= 255
print("shape：", image_bgr.shape)
print("dtype：", image_bgr.dtype)
print("范围：", int(image_bgr.min()), int(image_bgr.max()))
```

`shape=(H,W,C)` 表示高、宽、通道。它不是“六张图、每张八行、三个特征”，而是一幅有二维邻接关系的图像。

<QuizBlock question="一个彩色 OpenCV 数组的 shape 是 (480, 640, 3)，其中 480 表示什么？" :options="['图像高度，也就是行数', '图像宽度', '批次大小', '颜色位深']" :answer="0" explanation="OpenCV/NumPy 彩色图像通常采用 HWC：高度、宽度、通道数。" />

## 2. BGR、RGB 与灰度不是显示偏好

OpenCV 默认彩色顺序是 BGR，Matplotlib 和多数深度学习数据接口通常按 RGB 解释。若直接把 BGR 交给 RGB 显示器，红色和蓝色会交换。

```python
image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

assert image_rgb.shape == image_bgr.shape
assert image_gray.shape == image_bgr.shape[:2]
assert np.array_equal(image_rgb[..., 0], image_bgr[..., 2])
print("BGR 像素：", image_bgr[2, 3].tolist())
print("RGB 像素：", image_rgb[2, 3].tolist())
print("灰度 shape：", image_gray.shape)
```

灰度转换不是简单删除两个通道，而是按亮度贡献组合颜色。灰度图通常是二维数组；不要为了“统一”而无意义地复制成三个相同通道。

```python
# tutorial-validation: syntax-only
import matplotlib.pyplot as plt

figure, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].imshow(image_rgb)
axes[0].set_title("RGB")
axes[1].imshow(image_gray, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("灰度")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
```

<div class="ds-warning"><strong>颜色正确不代表合同正确</strong>一幅灰度纹理图即使把 BGR 与 RGB 搞反也可能看不出异常。必须检查通道语义，而不是靠肉眼猜。</div>

<QuizBlock question="用 Matplotlib 显示 cv.imread 得到的彩色图之前，通常需要哪一步？" :options="['BGR 转 RGB', 'RGB 转 HSV', '把宽和高交换', '删除 dtype']" :answer="0" explanation="OpenCV 默认读取为 BGR，而 Matplotlib 按 RGB 解释三通道数组。" />

## 3. 行列索引与 x、y 坐标顺序相反

NumPy 用 `[row, column]` 访问数组，图像几何和 OpenCV 点坐标常写成 `(x, y)`。对应关系是：

$$
\text{array}[y,x]\quad\longleftrightarrow\quad\text{point}(x,y).
$$

```python
row, column = 2, 3
b, g, r = image_bgr[row, column]
x, y = column, row

assert image_bgr[y, x].tolist() == [190, 92, 38]
print("数组索引 [row, column]：", [row, column])
print("图像坐标 (x, y)：", [x, y])
print("BGR：", [int(b), int(g), int(r)])
```

<PixelProbeLab />

把横纵坐标写反，程序往往不会报错，只会在错误位置裁图、画框或读取像素。这类错误比语法错误更危险。

## 4. ROI 是数组切片，不是复制出来的新对象

Region of Interest（ROI）通常用切片获得：

```python
roi_view = image_bgr[1:5, 2:6]
roi_copy = image_bgr[1:5, 2:6].copy()

assert roi_view.shape == (4, 4, 3)
assert np.shares_memory(roi_view, image_bgr)
assert not np.shares_memory(roi_copy, image_bgr)
print("ROI shape：", roi_view.shape)
```

切片通常共享底层内存。修改 `roi_view` 会改变原图；需要独立样本时应显式 `.copy()`。批量裁 patch 时尤其要明确这一点。

<QuizBlock question="为什么训练样本裁剪后常调用 .copy()？" :options="['避免裁剪视图与原数组共享内存造成意外修改', '让图片自动变成 RGB', '提高图像分辨率', '自动生成标签']" :answer="0" explanation="NumPy 切片通常是 view；显式复制能冻结独立样本边界。" />

## 5. 文件读取必须验证失败和解码结果

`cv.imread` 读取失败时通常返回 `None`。检查文件是否存在还不够：路径存在但内容损坏、格式不受支持，同样可能解码失败。

```python
# tutorial-validation: syntax-only
from pathlib import Path

image_path = Path("data/inspection/frame_0001.png")
if not image_path.is_file():
    raise FileNotFoundError(image_path)

loaded_bgr = cv.imread(str(image_path), cv.IMREAD_COLOR)
if loaded_bgr is None:
    raise ValueError(f"无法解码图像：{image_path}")

print(loaded_bgr.shape, loaded_bgr.dtype)
```

Notebook 中用 Matplotlib 显示，课程不使用 OpenCV GUI 窗口。保存文件时要检查 `cv.imwrite` 的布尔返回值，不要默认写入成功。

### PNG 与 JPEG 保存的不是同一种承诺

PNG 对像素值做无损压缩；JPEG 通常有损，会在边缘和细纹理附近产生变化。

```python
png_ok, png_buffer = cv.imencode(".png", image_bgr)
jpeg_ok, jpeg_buffer = cv.imencode(".jpg", image_bgr, [cv.IMWRITE_JPEG_QUALITY, 80])
assert png_ok and jpeg_ok

png_roundtrip = cv.imdecode(png_buffer, cv.IMREAD_COLOR)
jpeg_roundtrip = cv.imdecode(jpeg_buffer, cv.IMREAD_COLOR)
assert np.array_equal(image_bgr, png_roundtrip)

jpeg_max_error = np.abs(image_bgr.astype(np.int16) - jpeg_roundtrip.astype(np.int16)).max()
print("PNG 字节数：", len(png_buffer))
print("JPEG 字节数：", len(jpeg_buffer))
print("JPEG 最大像素差：", int(jpeg_max_error))
```

对照片分类，适度 JPEG 压缩可能可接受；对细小缺陷、掩码和像素级测量，压缩伪影可能直接改变标签边界。二值 mask 不应保存成有损 JPEG。

<QuizBlock question="像素级分割 mask 更适合保存成哪种格式？" :options="['无损 PNG', '有损 JPEG', '任意格式都完全等价', '只保存在屏幕截图中']" :answer="0" explanation="类别编号和边界必须精确保留，有损 JPEG 会引入新的像素值和边缘伪影。" />

## 6. 从 HWC、uint8 到 NCHW、float32

OpenCV 返回的常见图像是 `HWC`、BGR、`uint8`。深度学习框架常接收 `NCHW`、RGB、`float32`，并要求明确的归一化规则。

```python
image_float = image_rgb.astype(np.float32) / 255.0
image_chw = np.transpose(image_float, (2, 0, 1))
image_batch = image_chw[None, ...]

assert image_float.dtype == np.float32
assert 0.0 <= image_float.min() <= image_float.max() <= 1.0
assert image_chw.shape == (3, 6, 8)
assert image_batch.shape == (1, 3, 6, 8)
print("模型批次 shape：", image_batch.shape)
```

除以 255 只是一个具体合同，不是所有图像的普遍规则。若输入已经是 `0–1` 浮点数，再除一次就会把信号缩小 255 倍；若原始图像是 12/16 位，也不能照搬。

## 7. 在系统边界验证图像合同

```python
def validate_bgr_uint8(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 HWC 三通道图像，实际为 {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"期望 uint8，实际为 {image.dtype}")
    if image.size == 0:
        raise ValueError("图像不能为空")
    return image

validated = validate_bgr_uint8(image_bgr)
assert validated is image_bgr

try:
    validate_bgr_uint8(image_gray)
except ValueError as error:
    print("捕获预期错误：", error)
```

内部处理函数可以依赖已经验证的结构，避免每一行代码都重复猜测输入。

<div class="ds-stop"><strong>停下来想一想</strong>某批图像中一半是 `uint8 0–255`，另一半是 `float32 0–1`。如果直接拼成一个数组，NumPy 可能把全部数据提升成浮点数，但这并没有自动统一语义。你会在哪个边界阻止它？</div>

## 8. 知识检查

<QuizBlock question="OpenCV 图像数组中的 image[20, 35] 对应哪个图像坐标？" :options="['(x=35, y=20)', '(x=20, y=35)', '(x=20, y=20)', '(x=35, y=35)']" :answer="0" explanation="数组先行后列，即 [y,x]；点坐标通常写成 (x,y)。" />

<QuizBlock question="一幅图像的 dtype 是 uint8，执行 250+20 最需要警惕什么？" :options="['整数溢出或饱和语义不明确', '通道自动增加', '宽度变为 270', '文件自动删除']" :answer="0" explanation="uint8 不能表示 270；应先提升类型、计算、裁剪，再显式转换。" />

## 9. 代码练习：生成图像审计记录

实现 `inspect_image`，返回图像的 `height`、`width`、`channels`、`dtype`、最小值、最大值和每通道均值。函数同时接受二维灰度图和三维彩色图，拒绝空数组与其他维度。

<details><summary>提示</summary>

二维图的通道数记为 1；彩色图沿 `(0,1)` 两个空间轴求均值。

</details>

<details><summary>参考实现</summary>

```python
def inspect_image(image):
    image = np.asarray(image)
    if image.size == 0:
        raise ValueError("图像不能为空")
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
        channel_mean = [float(image.mean())]
    elif image.ndim == 3:
        height, width, channels = image.shape
        channel_mean = image.mean(axis=(0, 1)).astype(float).tolist()
    else:
        raise ValueError(f"只接受 HW 或 HWC，实际为 {image.shape}")
    return {
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": str(image.dtype),
        "minimum": float(image.min()),
        "maximum": float(image.max()),
        "channel_mean": channel_mean,
    }

audit = inspect_image(image_bgr)
assert audit["height"] == 6 and audit["width"] == 8
assert audit["channels"] == 3 and audit["dtype"] == "uint8"
print(audit)
```

</details>

## 10. 本讲小结

- 数字图像是经过采样和量化的空间数组，必须同时声明形状、通道、类型和范围。
- OpenCV 常用 BGR/HWC，显示和模型接口常需要显式转换为 RGB 或 CHW。
- NumPy 索引是 `[row,column]=[y,x]`，OpenCV 点坐标常写 `(x,y)`。
- ROI、文件解码和有损压缩都会改变数据合同，必须在系统边界检查。
- 后续像素、几何和空间操作都建立在这份合同上。

## 参考资料

- [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)：Python 安装、核心操作与图像处理入口。
- [Basic Operations on Images](https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html)：像素、属性、ROI、通道与边界操作。
- [Image file reading and writing](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html)：`imread`、`imwrite`、`imencode` 与 `imdecode` API。
- 本页合成图像、正文、代码、问题和像素探针实验均为本课程原创，只使用 OpenCV 与 NumPy 的公共 API。

<DsLessonProgress lesson-id="V1" next-label="V2：像素与几何变换" next-href="/courses/data-science/v2-pixel-geometric-transformations" />
