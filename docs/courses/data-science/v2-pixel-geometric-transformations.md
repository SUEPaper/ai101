---
course: data-science
title: 像素变换与几何变换
description: 用 OpenCV 与 NumPy 学习亮度、对比度、Gamma、直方图、缩放、翻转、旋转、插值和标注同步。
outline: [2, 3]
---

<DsLessonHero
  number="V2"
  title="像素变换与几何变换"
  eyebrow="像素变换改变数值，几何变换改变取样位置；两者都可能改变模型看到的证据"
  duration="建议 130–170 分钟"
  :outcomes="[
    '用浮点中间结果安全实现亮度、对比度与 Gamma 变换',
    '解释直方图能显示什么、不能显示什么',
    '根据图像、mask 和标签语义选择插值方法',
    '同步变换图像与边界框，并记录可复现的增强参数'
  ]"
/>

## 本讲要解决的问题

训练集来自两台相机：一台整体偏暗，另一台分辨率更高。团队把所有图像统一“拉亮并缩成 224×224”，结果细小缺陷消失，边界框又与图像错位。问题不在 API 是否调用成功，而在每个变换保留了什么、丢失了什么，以及标签有没有使用同一套几何参数。

```python
import cv2 as cv
import numpy as np

rng = np.random.default_rng(20260824)
scene = np.full((72, 96, 3), [34, 48, 58], dtype=np.uint8)
cv.rectangle(scene, (24, 18), (67, 51), (48, 148, 224), thickness=-1)
cv.line(scene, (29, 44), (61, 25), (205, 214, 225), thickness=2)
noise = rng.normal(0, 5, scene.shape).astype(np.int16)
scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)
bbox_xywh = np.array([24, 18, 44, 34], dtype=np.int32)

assert scene.shape == (72, 96, 3)
assert bbox_xywh.tolist() == [24, 18, 44, 34]
```

## 1. 点变换只看当前像素

点变换可以写成 $s=T(r)$：输出像素 `s` 只取决于同一位置的输入 `r`。亮度、对比度、反相和 Gamma 都属于这一类。

### 1.1 先提升类型，再计算、裁剪和转换

```python
def adjust_linear(image, alpha=1.0, beta=0.0):
    working = image.astype(np.float32)
    adjusted = alpha * working + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)

brighter = adjust_linear(scene, alpha=1.12, beta=18)
lower_contrast = adjust_linear(scene, alpha=0.65, beta=35)

assert brighter.dtype == np.uint8
assert brighter.shape == scene.shape
print("原均值：", round(float(scene.mean()), 2))
print("变换后均值：", round(float(brighter.mean()), 2))
```

不要直接在 `uint8` 上执行可能越界的算术。正确顺序是：提升为浮点数或更宽整数，运算，`clip`，最后显式转回。

### 1.2 Gamma 改变暗部和亮部的分配

归一化输入 $r\in[0,1]$ 时，一种常见形式为：

$$
s=r^{1/\gamma}.
$$

```python
def gamma_lut(gamma):
    if gamma <= 0:
        raise ValueError("gamma 必须大于 0")
    values = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(values ** (1.0 / gamma) * 255, 0, 255).astype(np.uint8)

gamma_bright = cv.LUT(scene, gamma_lut(1.8))
gamma_dark = cv.LUT(scene, gamma_lut(0.65))

assert gamma_bright.mean() > scene.mean()
assert gamma_dark.mean() < scene.mean()
print("Gamma 1.8 均值：", round(float(gamma_bright.mean()), 2))
```

Gamma 公式在不同系统中可能使用互为倒数的记号。代码、实验记录和部署配置必须写清具体公式，不能只写“gamma=1.8”。

<QuizBlock question="为什么不建议直接在 uint8 图像上执行任意加减乘法？" :options="['可能溢出或绕回，结果不再代表预期亮度', '会自动增加通道', '图像一定变成灰度', 'OpenCV 不支持数组']" :answer="0" explanation="uint8 只有 0–255；应先提升类型，计算后裁剪并显式转换。" />

## 2. 直方图统计强度，不保留空间位置

```python
gray = cv.cvtColor(scene, cv.COLOR_BGR2GRAY)
histogram = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel()

assert int(histogram.sum()) == gray.size
dominant_value = int(histogram.argmax())
print("像素数：", gray.size)
print("出现最多的灰度：", dominant_value)
```

直方图能回答亮暗范围、饱和比例、是否出现多峰等问题，却不能告诉你高亮像素位于左上角还是目标内部。两幅空间结构完全不同的图，可能有相同直方图。

```python
shuffled = gray.ravel()[rng.permutation(gray.size)].reshape(gray.shape)
shuffled_histogram = cv.calcHist([shuffled], [0], None, [256], [0, 256]).ravel()
assert np.array_equal(histogram, shuffled_histogram)
assert not np.array_equal(gray, shuffled)
print("直方图相同，空间结构相同：", np.array_equal(gray, shuffled))
```

<QuizBlock question="两幅图的灰度直方图完全相同，可以推出什么？" :options="['各灰度值的计数相同，但空间排列可能完全不同', '两幅图逐像素相同', '物体位置相同', '边缘数量一定相同']" :answer="0" explanation="直方图丢弃空间坐标，只保留强度频数。" />

## 3. 缩放同时涉及几何和重采样

```python
small = cv.resize(scene, (48, 36), interpolation=cv.INTER_AREA)
restored = cv.resize(small, (96, 72), interpolation=cv.INTER_LINEAR)

assert small.shape == (36, 48, 3)
assert restored.shape == scene.shape
roundtrip_mae = np.abs(scene.astype(np.int16) - restored.astype(np.int16)).mean()
print("缩小再放大 MAE：", round(float(roundtrip_mae), 2))
```

缩小会丢失高频细节，之后再放大不能恢复原始像素。一般经验：

- 照片缩小常用 `INTER_AREA`。
- 照片放大常用 `INTER_LINEAR` 或更慢的 `INTER_CUBIC`。
- 类别 mask 使用 `INTER_NEAREST`，避免制造不存在的类别编号。

```python
class_mask = np.zeros((4, 4), dtype=np.uint8)
class_mask[:, 2:] = 2
nearest_mask = cv.resize(class_mask, (11, 11), interpolation=cv.INTER_NEAREST)
linear_mask = cv.resize(class_mask, (11, 11), interpolation=cv.INTER_LINEAR)

print("最近邻类别：", np.unique(nearest_mask).tolist())
print("双线性产生的值：", np.unique(linear_mask).tolist())
assert set(np.unique(nearest_mask)) == {0, 2}
assert 1 in np.unique(linear_mask)
```

<QuizBlock question="把类别为 0 和 2 的分割 mask 放大时，为什么应优先最近邻插值？" :options="['避免插值产生不存在的类别 1', '让图像更模糊', '把 mask 转成 RGB', '最近邻一定最适合照片']" :answer="0" explanation="mask 存的是离散标签，不是可连续平均的亮度。" />

## 4. 裁剪、填充和长宽比定义视野

直接把任意图像压成固定正方形会改变物体比例。常见策略各有代价：

- 拉伸：保留全部视野，但形状被扭曲。
- 等比例缩放后填充：保留比例，但引入边界区域。
- 中心裁剪：不引入填充，但可能切掉边缘目标。
- 滑窗或重叠 patch：保留小目标分辨率，但可能让同一对象跨集合泄漏。

```python
top, bottom, left, right = 8, 8, 12, 12
padded = cv.copyMakeBorder(
    scene,
    top,
    bottom,
    left,
    right,
    borderType=cv.BORDER_CONSTANT,
    value=(0, 0, 0),
)

shifted_bbox = bbox_xywh.copy()
shifted_bbox[:2] += [left, top]
assert padded.shape == (88, 120, 3)
assert shifted_bbox.tolist() == [36, 26, 44, 34]
```

填充后图像原点不变，但原目标在新画布中的坐标移动了。标签必须同步加上左、上填充值。

## 5. 翻转图像时也要翻转边界框

边界框记为 `(x,y,w,h)`。宽度为 `W` 的图像水平翻转后：

$$
x'=W-x-w.
$$

```python
def horizontal_flip_with_box(image, box_xywh):
    image = np.asarray(image)
    box = np.asarray(box_xywh, dtype=np.int32).copy()
    flipped = cv.flip(image, 1)
    box[0] = image.shape[1] - box[0] - box[2]
    return flipped, box

flipped_scene, flipped_bbox = horizontal_flip_with_box(scene, bbox_xywh)
assert flipped_scene.shape == scene.shape
assert flipped_bbox.tolist() == [28, 18, 44, 34]
print("翻转后边界框：", flipped_bbox.tolist())
```

<GeometryTransformLab />

只翻图不翻标签，代码仍然运行，训练却会把目标位置当成背景、把背景当成目标。

<QuizBlock question="图像宽 100，边界框为 x=20、w=30。水平翻转后的 x 是多少？" :options="['50', '80', '20', '30']" :answer="0" explanation="x'=W-x-w=100-20-30=50。" />

## 6. 旋转是逆向采样，不只是移动像素

```python
height, width = scene.shape[:2]
center = (width / 2, height / 2)
rotation_matrix = cv.getRotationMatrix2D(center, angle=12, scale=1.0)
rotated = cv.warpAffine(
    scene,
    rotation_matrix,
    (width, height),
    flags=cv.INTER_LINEAR,
    borderMode=cv.BORDER_REFLECT_101,
)

assert rotation_matrix.shape == (2, 3)
assert rotated.shape == scene.shape
print(rotation_matrix.round(3))
```

目标画布上的每个像素会反查原图中的来源坐标，再通过插值获得数值。边界之外如何填充也是变换合同的一部分。检测框若旋转，简单的轴对齐框通常还要包住旋转后的四个角；分割 mask 则必须用相同矩阵和最近邻插值。

## 7. 数据增强必须受任务语义约束

“增强越多越好”是错误原则。每种增强都隐含标签不变或标签等变假设：

| 任务 | 水平翻转 | 垂直翻转 | 任意旋转 |
|---|---|---|---|
| 自然物体分类 | 常常可用 | 需判断 | 小角度常见 |
| 文本/OCR | 常常改变语义 | 改变语义 | 仅允许真实倾斜范围 |
| 遥感俯视图 | 常常可用 | 常常可用 | 视方向语义而定 |
| 工业固定工位 | 可能破坏绝对位置 | 可能不可能发生 | 服从相机和工装几何 |

随机增强还应记录随机种子、参数范围和实现版本。训练集可以随机增强；验证集和测试集应使用冻结、可复查的确定性预处理。

<div class="ds-warning"><strong>预处理参数也可能泄漏</strong>若根据全部数据的直方图选择 Gamma、裁剪阈值或颜色校正参数，测试集信息已经进入流程。先按原始实体分组切分，再只用训练集确定参数。</div>

<QuizBlock question="检测任务随机翻转时，正确的最小合同是什么？" :options="['图像与边界框共享同一次随机参数', '只翻图像', '只翻边界框', '每个框单独随机翻转']" :answer="0" explanation="输入和空间标签必须执行同一几何变换。" />

## 8. 代码练习：等比例缩放并填充

实现 `letterbox_with_box`：把图像等比例缩放到目标画布内，再居中填充，同时更新一个 `(x,y,w,h)` 边界框。照片使用线性插值；函数返回图像、边界框和 `(scale,left,top)`。

<details><summary>提示</summary>

比例取 `min(target_width/width, target_height/height)`；边界框四个量先乘比例，再给 `x,y` 加左、上填充。

</details>

<details><summary>参考实现</summary>

```python
def letterbox_with_box(image, box_xywh, target_size=(128, 128)):
    target_width, target_height = target_size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)

    left = (target_width - new_width) // 2
    right = target_width - new_width - left
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    output = cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

    box = np.asarray(box_xywh, dtype=np.float32).copy()
    box *= scale
    box[:2] += [left, top]
    return output, box, (scale, left, top)

boxed, transformed_bbox, transform = letterbox_with_box(scene, bbox_xywh)
assert boxed.shape == (128, 128, 3)
assert np.all(transformed_bbox >= 0)
print("变换：", transform)
print("边界框：", transformed_bbox.round(2).tolist())
```

</details>

## 9. 本讲小结

- 点变换改变像素数值；几何变换改变空间采样位置。
- `uint8` 运算要经过类型提升、计算、裁剪和显式转换。
- 直方图保留强度计数但丢弃空间结构。
- 图像、类别 mask、边界框和关键点需要不同插值语义，但必须共享同一几何参数。
- 增强策略必须由数据生成过程决定，并只在训练集上调节。

## 参考资料

- [Changing Colorspaces](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)：颜色空间转换与通道范围。
- [Geometric Transformations of Images](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)：缩放、平移、旋转、仿射和透视变换。
- [Geometric Image Transformations API](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)：映射、插值与边界的精确定义。
- [Histograms in OpenCV](https://docs.opencv.org/4.x/de/db2/tutorial_py_table_of_contents_histograms.html)：直方图计算与解释入口。
- 本页场景、图像数组、正文、代码、问题和变换坐标实验均为本课程原创，只使用 OpenCV 与 NumPy 的公共 API。

<DsLessonProgress lesson-id="V2" next-label="V3：空间操作与区域测量" next-href="/courses/data-science/v3-spatial-segmentation" />
