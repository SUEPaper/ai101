---
course: data-science
title: 空间操作、分割与区域测量
description: 用 OpenCV 学习卷积核、平滑、梯度、边缘、阈值、形态学、连通区域、轮廓和批量特征表。
outline: [2, 3]
---

<DsLessonHero
  number="V3"
  title="空间操作、分割与区域测量"
  eyebrow="邻域操作把局部结构变成信号，分割再把像素集合变成可以统计的对象"
  duration="建议 150–190 分钟"
  :outcomes="[
    '解释卷积核、边界和局部邻域怎样改变输出',
    '区分高斯、中值、梯度与 Canny 的作用',
    '组合阈值和形态学获得可测量的二值区域',
    '把连通区域和轮廓测量整理成可审计的 DataFrame'
  ]"
/>

## 本讲要解决的问题

一批灰度质检图包含亮色零件、椒盐噪点、非均匀背景和细小间隙。我们不急着训练神经网络，而是先建立一个可解释基线：去噪、分割、清理 mask、计算每个对象的面积与位置，并把结果送入熟悉的数据科学流程。

```python
import cv2 as cv
import numpy as np
import pandas as pd

rng = np.random.default_rng(20260824)
height, width = 96, 128
row_grid, column_grid = np.mgrid[:height, :width]
background = 34 + 0.18 * column_grid + 0.08 * row_grid
gray = background.astype(np.float32)
cv.rectangle(gray, (18, 20), (52, 66), 182, thickness=-1)
cv.circle(gray, (91, 47), 19, 214, thickness=-1)
cv.line(gray, (33, 42), (47, 42), 78, thickness=2)
gray += rng.normal(0, 6, gray.shape)
gray = np.clip(gray, 0, 255).astype(np.uint8)

salt_rows = rng.integers(0, height, 70)
salt_columns = rng.integers(0, width, 70)
gray[salt_rows, salt_columns] = 255

assert gray.shape == (96, 128)
assert gray.dtype == np.uint8
```

## 1. 空间操作读取一个邻域

点变换只看当前像素；空间操作会把当前位置附近的值组合起来。以 `3×3` 均值核为例：

$$
K=\frac{1}{9}
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}.
$$

```python
mean_kernel = np.ones((3, 3), dtype=np.float32) / 9
mean_filtered = cv.filter2D(gray, ddepth=-1, kernel=mean_kernel, borderType=cv.BORDER_REFLECT_101)

assert mean_filtered.shape == gray.shape
assert mean_filtered.dtype == gray.dtype
print("原图标准差：", round(float(gray.std()), 2))
print("均值滤波标准差：", round(float(mean_filtered.std()), 2))
```

`ddepth=-1` 表示输出使用输入类型。边界像素没有完整邻域，所以 `borderType` 会改变结果；它不是无关实现细节。

<QuizBlock question="3×3 核处理图像边缘时为什么必须定义 borderType？" :options="['边缘缺少完整邻域，需要规定图像外的值怎样获得', '为了自动增加颜色通道', '因为中心像素不存在', '只为改变文件格式']" :answer="0" explanation="卷积核会访问图像外侧，填零、复制和反射会产生不同边界响应。" />

## 2. 高斯和中值滤波保留的信息不同

```python
gaussian = cv.GaussianBlur(gray, (5, 5), sigmaX=1.2, borderType=cv.BORDER_REFLECT_101)
median = cv.medianBlur(gray, 5)

bright_noise_before = int((gray == 255).sum())
bright_noise_gaussian = int((gaussian >= 250).sum())
bright_noise_median = int((median >= 250).sum())
print("255 噪点数：", bright_noise_before)
print("高斯后 >=250：", bright_noise_gaussian)
print("中值后 >=250：", bright_noise_median)
```

- 高斯滤波是加权平均，适合平滑近似高斯噪声，但会软化边缘。
- 中值滤波取邻域中位数，对孤立椒盐噪点更稳健，但核过大会吞掉细结构。
- “去噪”从来不是免费操作；它总会牺牲部分细节或位置精度。

<QuizBlock question="对少量孤立的 0/255 椒盐噪点，哪种滤波通常更有针对性？" :options="['中值滤波', '任意大核均值滤波', '只提高 JPEG 压缩', '交换 RGB 通道']" :answer="0" explanation="中值对孤立极端值稳健，同时通常比均值更能保留边缘。" />

## 3. 梯度和边缘不是对象本身

Sobel 近似计算水平和垂直方向导数；梯度幅值在强度快速变化处较大。

```python
sobel_x = cv.Sobel(gaussian, cv.CV_32F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gaussian, cv.CV_32F, 0, 1, ksize=3)
gradient_magnitude = cv.magnitude(sobel_x, sobel_y)
edges = cv.Canny(gaussian, threshold1=45, threshold2=110)

assert gradient_magnitude.dtype == np.float32
assert set(np.unique(edges)).issubset({0, 255})
print("最大梯度：", round(float(gradient_magnitude.max()), 2))
print("Canny 边缘像素：", int((edges > 0).sum()))
```

梯度图表示变化强度，Canny 输出细化后的候选边缘。它们没有自动回答“哪个封闭区域是对象”“边缘属于哪种缺陷”或“对象内部是什么类别”。

<QuizBlock question="Canny 输出的白色像素最准确的解释是什么？" :options="['满足算法条件的候选边缘位置', '已经识别出的对象类别', '所有前景内部像素', '图像的真实物理边界保证']" :answer="0" explanation="边缘检测输出局部强度变化证据，还需要区域、任务语义和验证。" />

## 4. 阈值把连续强度变成二值决策

固定阈值要求你预先给定切分位置；Otsu 方法在当前图像直方图上选择一个使类内方差较小的阈值。

```python
otsu_threshold, raw_mask = cv.threshold(
    gaussian,
    0,
    255,
    cv.THRESH_BINARY + cv.THRESH_OTSU,
)

assert raw_mask.dtype == np.uint8
assert set(np.unique(raw_mask)).issubset({0, 255})
print("Otsu 阈值：", round(float(otsu_threshold), 2))
print("前景比例：", round(float((raw_mask > 0).mean()), 3))
```

Otsu 并不知道对象是什么。背景强烈不均匀、前景占比极小或直方图不呈双峰时，它可能失败。自适应阈值能处理局部照明变化，但窗口大小和常数同样是需要验证的参数。

```python
adaptive_mask = cv.adaptiveThreshold(
    gaussian,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    blockSize=21,
    C=5,
)
assert adaptive_mask.shape == gray.shape
print("自适应阈值前景比例：", round(float((adaptive_mask > 0).mean()), 3))
```

<QuizBlock question="Otsu 阈值是否保证得到业务上正确的对象 mask？" :options="['不保证，它只根据当前强度分布优化统计准则', '保证', '只有彩色图保证', '只要像素多就保证']" :answer="0" explanation="统计分离不等于语义正确，仍需与标注和失败切片比较。" />

## 5. 形态学用结构元素修改形状

二值形态学把前景看成像素集合，再用结构元素探测邻域。

```python
structure = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
opened = cv.morphologyEx(raw_mask, cv.MORPH_OPEN, structure, iterations=1)
clean_mask = cv.morphologyEx(opened, cv.MORPH_CLOSE, structure, iterations=2)

assert structure.shape == (3, 3)
print("原始前景像素：", int((raw_mask > 0).sum()))
print("开闭后前景像素：", int((clean_mask > 0).sum()))
```

- 腐蚀缩小白色前景。
- 膨胀扩大白色前景。
- 开运算先腐蚀后膨胀，常用于去除小白噪点。
- 闭运算先膨胀后腐蚀，常用于填补小黑孔或连接窄间隙。

结构元素的形状、大小和迭代次数都表达了物理尺度假设。若图像分辨率从 `0.1 mm/pixel` 变成 `0.02 mm/pixel`，同一个 `3×3` 核不再对应同样的真实尺寸。

<SegmentationMeasureLab />

<QuizBlock question="开运算最典型的组合顺序是什么？" :options="['先腐蚀后膨胀', '先膨胀后腐蚀', '两次 Canny', '先 RGB 后 BGR']" :answer="0" explanation="开运算常用于去除小前景噪点；闭运算的顺序相反。" />

## 6. 连通区域把 mask 变成对象表

`connectedComponentsWithStats` 会为每个连通区域分配编号，并返回轴对齐外接框、面积和质心。编号 0 是背景。

```python
component_count, labels, stats, centroids = cv.connectedComponentsWithStats(
    clean_mask,
    connectivity=8,
)

rows = []
for component_id in range(1, component_count):
    x, y, box_width, box_height, area = stats[component_id]
    center_x, center_y = centroids[component_id]
    rows.append({
        "component_id": component_id,
        "x": int(x),
        "y": int(y),
        "width": int(box_width),
        "height": int(box_height),
        "area_px": int(area),
        "centroid_x": float(center_x),
        "centroid_y": float(center_y),
    })

component_table = pd.DataFrame(rows)
assert len(component_table) == component_count - 1
assert (component_table["area_px"] > 0).all()
print(component_table.round(2))
```

这一步把像素集合转换成了熟悉的表格数据：现在可以画面积分布、按批次比较、设定复核阈值或与人工标注连接。

### 物理单位需要空间标定

若每像素代表 `0.08 mm × 0.08 mm`，区域物理面积为：

```python
millimeter_per_pixel = 0.08
component_table["area_mm2"] = component_table["area_px"] * millimeter_per_pixel ** 2
print(component_table[["component_id", "area_px", "area_mm2"]].round(3))
```

像素面积不是跨相机可比较的物理量。空间标定、缩放历史和裁剪协议必须进入数据卡。

## 7. 轮廓提供边界级测量

```python
contours, hierarchy = cv.findContours(clean_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour_rows = []
for contour_id, contour in enumerate(contours, start=1):
    x, y, box_width, box_height = cv.boundingRect(contour)
    contour_rows.append({
        "contour_id": contour_id,
        "area_px": float(cv.contourArea(contour)),
        "perimeter_px": float(cv.arcLength(contour, closed=True)),
        "x": x,
        "y": y,
        "width": box_width,
        "height": box_height,
    })

contour_table = pd.DataFrame(contour_rows)
assert len(contour_table) >= 1
print(contour_table.round(2))
```

连通区域面积按白色像素计数，`contourArea` 按轮廓几何计算，边界离散化时两者可能不完全相同。必须给指标写明定义，不能都笼统叫“面积”。

<div class="ds-warning"><strong>预处理会改写真值</strong>开闭运算让 mask 更整齐，同时也可能合并相邻对象、填掉真实孔洞或删除微小缺陷。发布结果时要保存原图、原始 mask、处理参数与最终测量，不能只保留漂亮的输出。</div>

## 8. 批处理函数要返回证据而不只返回图片

```python
def measure_bright_regions(image_gray, minimum_area=80):
    image_gray = np.asarray(image_gray)
    if image_gray.ndim != 2 or image_gray.dtype != np.uint8:
        raise ValueError("期望 uint8 灰度图")

    filtered = cv.medianBlur(image_gray, 5)
    threshold_value, binary = cv.threshold(
        filtered, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    count, _, stats, centers = cv.connectedComponentsWithStats(binary, connectivity=8)

    records = []
    for component_id in range(1, count):
        x, y, box_width, box_height, area = stats[component_id]
        if area < minimum_area:
            continue
        records.append({
            "component_id": component_id,
            "threshold": float(threshold_value),
            "x": int(x),
            "y": int(y),
            "width": int(box_width),
            "height": int(box_height),
            "area_px": int(area),
            "centroid_x": float(centers[component_id, 0]),
            "centroid_y": float(centers[component_id, 1]),
        })
    return binary, pd.DataFrame(records)

measured_mask, measurements = measure_bright_regions(gray)
assert measured_mask.shape == gray.shape
assert len(measurements) >= 2
print(measurements.round(2))
```

真实批处理还应给每行添加 `image_id`、设备、采集时间、处理版本和参数。若函数只输出一张处理后图片，就无法追踪测量来自哪个原始样本。

## 9. 从经典基线走向深度学习

这条工作流不会被深度学习完全替代：

- 用于发现坏帧、饱和、噪声和相机漂移。
- 作为标注预处理或弱标签候选，但必须人工复核。
- 提供可解释基线，判断复杂模型是否真正增加价值。
- 帮助理解 CNN 低层卷积为何响应边缘和纹理。
- 在分割模型输出后清理小区域、计算业务对象指标。

边界也要明确：固定阈值依赖成像稳定性，手工特征对复杂外观变化脆弱，形态学规则不会自动理解语义。

<div class="ds-stop"><strong>停下来想一想</strong>验证集 IoU 提高，但形态学闭运算把两个相邻缺陷合成一个对象，导致对象计数下降。哪个指标更接近最终行动？答案取决于任务是像素覆盖、缺陷计数还是停线决策，不能只保留一个分数。</div>

## 10. 代码练习：输出可审计的区域报告

扩展 `measure_bright_regions`：为每个区域增加矩形填充率 `area/(width*height)`，并过滤掉填充率小于 0.25 的区域。返回表中还必须包含 `image_id` 和处理参数版本。

<details><summary>提示</summary>

先计算 `box_area=width*height`；将数据来源字段作为函数参数，而不是写死在函数内部。

</details>

<details><summary>参考实现</summary>

```python
def audited_region_report(image_gray, image_id, pipeline_version="opencv-v1"):
    _, table = measure_bright_regions(image_gray, minimum_area=80)
    if table.empty:
        return table.assign(
            image_id=pd.Series(dtype="string"),
            pipeline_version=pd.Series(dtype="string"),
            fill_ratio=pd.Series(dtype=float),
        )
    table = table.copy()
    table["fill_ratio"] = table["area_px"] / (table["width"] * table["height"])
    table = table.loc[table["fill_ratio"] >= 0.25].copy()
    table.insert(0, "image_id", image_id)
    table.insert(1, "pipeline_version", pipeline_version)
    return table.reset_index(drop=True)

report = audited_region_report(gray, "synthetic-frame-001")
assert {"image_id", "pipeline_version", "fill_ratio"}.issubset(report.columns)
assert (report["fill_ratio"] >= 0.25).all()
print(report.round(3))
```

</details>

## 11. 本讲小结

- 空间滤波把邻域按核与边界规则组合，平滑必然交换噪声抑制和细节保留。
- 梯度与 Canny 提供边缘证据，不自动产生对象语义。
- 阈值把连续强度变成决策；形态学按结构元素改变前景形状。
- 连通区域和轮廓把 mask 转成面积、质心、外接框与周长等表格特征。
- 原图、参数、处理版本和测量结果必须共同保存，才能形成可复查的数据科学证据。

## 参考资料

- [Image Processing in OpenCV](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)：平滑、阈值、形态学、梯度、边缘、轮廓和直方图入口。
- [Smoothing Images](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)：二维卷积、高斯、中值与双边滤波。
- [Morphological Transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)：腐蚀、膨胀、开闭和结构元素。
- [Contours: Getting Started](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)：轮廓查找、表示与绘制。
- 本页质检场景、合成图像、正文、代码、问题和分割测量实验均为本课程原创，只使用 OpenCV、NumPy 与 Pandas 的公共 API。

<DsLessonProgress lesson-id="V3" next-label="七次模块综合作业" next-href="/courses/data-science/assignments" />
