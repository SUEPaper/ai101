---
course: machine-learning
edition: industrial
title: 卷积神经网络（CNN）
description: 以冷轧钢带线扫表面缺陷分流为贯穿案例，从局部连接与参数共享推导二维互相关、输出几何、平移等变、感受野、池化与卷积反向传播，并用 NumPy、有限差分和可选 PyTorch 对照建立实现验收门禁。
outline: [2, 3]
---

<MlLessonHero
  number="15"
  title="卷积神经网络（CNN）"
  question="同一条细划痕可能出现在钢带图像的任何位置；怎样让模型只学习一组局部探测器，却能在整幅图上复用，并把每次复用产生的梯度完整收回来？"
  duration="建议 600–720 分钟"
  :outcomes="[
    '从全连接参数爆炸推导局部连接和参数共享，手算二维互相关、数学卷积、输出尺寸、参数量与计算量',
    '解释平移等变与不变、理论与有效感受野、步幅/填充/膨胀/池化的几何及采样代价',
    '实现单通道与多通道卷积、池化和卷积反向传播，并用有限差分及可选 PyTorch 对照验收梯度'
  ]"
/>

第 13 章建立了神经网络的表示能力，第 14 章建立了任意计算图的反向传播。本章假设已经完成数据科学课 [V1–V3 图像数据桥接模块](/courses/data-science/v1-digital-images)，不再讲 BGR/RGB、HWC/CHW、插值、阈值和形态学入门。本章把神经网络放进二维网格：图像的坐标不是普通特征编号，相邻像素、重复纹理和空间位置本身都含有结构。

卷积神经网络（convolutional neural network, CNN）的核心不是“图像必须用卷积”，而是三条可检验的建模假设：

1. **局部性**：一个低层视觉模式主要由邻域像素决定；
2. **空间共享**：同一种模式可以出现在不同位置，应使用同一组参数检测；
3. **层级组合**：边缘可组合成纹理，纹理再组合成零件或缺陷形态。

这三条是假设，不是自然定律。摄像机边缘畸变、位置相关照明、固定工装遮挡和绝对坐标规则都可能破坏空间共享；是否采用 CNN，必须从任务的数据生成过程而不是流行架构出发。

## 贯穿案例：冷轧钢带线扫表面缺陷分流

某冷轧产线在终检段使用线扫相机采集钢带表面灰度图。教学任务把 $16\times16$ 小块分成三类：

- `0 normal`：背景纹理，无需复核；
- `1 scratch`：细长划痕，送表面质量复核；
- `2 pit`：局部点蚀，送表面质量复核。

本章使用合成数据，不代表真实缺陷机理。真实上线至少要冻结下面的数据合同：

| 字段 | 合同示例 | 不冻结的后果 |
|---|---|---|
| 空间标定 | `0.08 mm / pixel`，横向与运行方向分别记录 | 同样 3×3 核对应的物理尺度漂移 |
| 灰度链路 | 曝光、增益、平场校正、位深、饱和率 | 模型把照明批次当缺陷 |
| 裁块协议 | 原始帧、重叠率、边界补齐、缺陷中心规则 | 同一缺陷切入多个集合造成泄漏 |
| 分组键 | `coil_id`，整卷只属于一个集合 | 随机切 patch 高估泛化 |
| 标签来源 | 复判方法、标注人、版本、争议规则 | 训练目标不可追溯 |
| 行动接口 | 自动放行、人工复核、停线属于不同风险级别 | 分类概率被误当控制指令 |

### 1. 先按钢卷分组，再做任何标准化或调参

下面生成 36 卷、每卷 24 个教学小块。卷级亮度漂移模拟批间变化；划痕位置和点蚀中心在小块内变化，迫使检测器学习“是什么”，而不是死记“在哪里”。

```python
import numpy as np

rng = np.random.default_rng(20260805)
coil_count = 36
patches_per_coil = 24
image_size = 16
coil_id = np.repeat(np.arange(coil_count), patches_per_coil)
sample_count = len(coil_id)
labels = np.empty(sample_count, dtype=np.int64)
images = np.empty((sample_count, 1, image_size, image_size), dtype=np.float64)

row_grid, column_grid = np.mgrid[:image_size, :image_size]
coil_offset = rng.normal(0.0, 0.055, size=coil_count)

for sample_index in range(sample_count):
    current_coil = coil_id[sample_index]
    label = (sample_index + current_coil) % 3
    labels[sample_index] = label
    texture = 0.18 + coil_offset[current_coil]
    texture += 0.018 * np.sin(column_grid * np.pi / 3)
    texture += rng.normal(0.0, 0.025, size=(image_size, image_size))

    if label == 1:  # 纵向细划痕，位置随机
        center = rng.integers(4, 12)
        texture[:, center] += rng.uniform(0.70, 0.95)
        texture[:, center + 1] += rng.uniform(0.20, 0.35)
    elif label == 2:  # 局部点蚀，中心随机
        center_row = rng.integers(4, 12)
        center_column = rng.integers(4, 12)
        radius_squared = (row_grid - center_row) ** 2 + (column_grid - center_column) ** 2
        texture -= rng.uniform(0.55, 0.75) * np.exp(-radius_squared / 2.2)

    images[sample_index, 0] = np.clip(texture, -0.8, 1.3)

coil_order = rng.permutation(coil_count)
train_coils = coil_order[:24]
validation_coils = coil_order[24:30]
test_coils = coil_order[30:]
train_mask = np.isin(coil_id, train_coils)
validation_mask = np.isin(coil_id, validation_coils)
test_mask = np.isin(coil_id, test_coils)

assert images.shape == (864, 1, 16, 16)
assert labels.shape == (864,)
assert set(train_coils).isdisjoint(validation_coils)
assert set(train_coils).isdisjoint(test_coils)
assert set(validation_coils).isdisjoint(test_coils)
assert train_mask.sum() == 576
assert validation_mask.sum() == test_mask.sum() == 144
assert np.isfinite(images).all()
print("训练/验证/测试：", train_mask.sum(), validation_mask.sum(), test_mask.sum())
print("三类样本数：", np.bincount(labels))
```

这里故意没有先把所有图像一起标准化。任何均值、方差、平场参数或数据增强策略都只能由训练卷确定，再原样应用到验证和测试卷。

<MlQuiz kind="数据合同" question="同一条长划痕被重叠裁成 8 个 patch，最可靠的切分方式是什么？" :options="['把 8 个 patch 随机分到三个集合', '先按 patch 标准化再切分', '按原始钢卷或缺陷事件分组，使相关 patch 只进入一个集合', '只保留其中像素最多的 patch']" :answer="2" explanation="重叠 patch 高度相关；按原始事件或钢卷分组才能阻断训练—测试泄漏。" />

## 2. 从全连接参数爆炸得到局部连接

若把 $H\times W\times C_{\text{in}}$ 图像展平成向量，再连接到 $H\times W\times C_{\text{out}}$ 个空间单元，权重数为

$$
HWC_{\text{in}}\cdot HWC_{\text{out}}.
$$

对 $64\times64\times3$ 输入和同分辨率的 16 个输出通道，这一层超过 $8.05\times10^8$ 个权重。更根本的问题是：左上角学到的划痕探测器与右下角的探测器没有任何参数关系。

卷积层做两次约束。

### 2.1 局部连接：稀疏权重矩阵

每个输出单元只查看 $K_h\times K_w$ 邻域，而不是整幅输入。若把卷积写成矩阵乘法，这个大矩阵绝大多数元素为 0。

### 2.2 参数共享：绑定重复权重

所有空间位置复用同一个核 $K$。若某个核元素在 3 000 个位置使用，它仍然只有一个参数；反向时，这 3 000 次使用的梯度贡献必须相加。

标准二维卷积层的参数量是

$$
\boxed{
C_{\text{out}}(C_{\text{in}}K_hK_w+1)
}
$$

其中每个输出通道只有一个共享偏置。参数量与 $H,W$ 无关，但计算量仍随输出位置数增长。

```python
def conv_budget(height, width, in_channels, out_channels, kernel_size):
    dense_outputs = height * width * out_channels
    dense_parameters = (height * width * in_channels + 1) * dense_outputs
    conv_parameters = out_channels * (in_channels * kernel_size ** 2 + 1)
    conv_macs = height * width * out_channels * in_channels * kernel_size ** 2
    return dense_parameters, conv_parameters, conv_macs

dense_parameters, conv_parameters, conv_macs = conv_budget(64, 64, 3, 16, 3)
assert dense_parameters == 805_371_904
assert conv_parameters == 448
assert conv_macs == 1_769_472
assert dense_parameters / conv_parameters > 1_000_000
print("全连接参数：", f"{dense_parameters:,}")
print("卷积参数：", f"{conv_parameters:,}")
print("卷积 MAC：", f"{conv_macs:,}")
```

卷积把“每处各学一套模式”改成“全图共享一套局部模板”。它显著降低样本复杂度和存储，却没有让每次推理免费：高分辨率特征图的乘加、激活和显存仍可能主导成本。

<MlQuiz kind="参数审计" question="输入尺寸从 64×64 增大到 128×128，通道数和 3×3 核不变。标准卷积层哪项不变？" :options="['参数量', '输出元素数', '乘加次数', '激活显存']" :answer="0" explanation="核参数在空间共享，所以不随 H、W 变化；计算和激活通常约增至四倍。" />

## 3. 框架所谓“卷积”通常是互相关

先考虑单通道、步幅 1、无填充。深度学习库常计算

$$
Y_{ij}
=\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1}
K_{uv}X_{i+u,j+v}.
$$

这在信号处理中严格称为**离散互相关**（cross-correlation），因为核没有翻转。数学卷积则先翻转两个坐标：

$$
(K*X)_{ij}
=\sum_{u,v}K_{K_h-1-u,K_w-1-v}X_{i+u,j+v}.
$$

对学习型核，两者只差一次参数重命名，表达能力相同；对 Sobel 这类固定方向核，翻转会改变响应符号或方向，因此实现文档必须说清。

### 3.1 手算一个 $2\times2$ 核

令

$$
X=
\begin{bmatrix}
1&2&3&4\\
5&6&7&8\\
9&10&11&12\\
13&14&15&16
\end{bmatrix},\qquad
K=
\begin{bmatrix}
1&0\\0&-1
\end{bmatrix}.
$$

左上角输出为

$$
Y_{00}=1\times1+2\times0+5\times0+6\times(-1)=-5.
$$

滑动后每个位置仍为 $-5$。若做数学卷积，翻转核得到相反的 $+5$。

```python
def cross_correlation_2d(x, kernel, stride=1, padding=0, dilation=1):
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kernel_height, kernel_width = kernel.shape
    effective_height = dilation * (kernel_height - 1) + 1
    effective_width = dilation * (kernel_width - 1) + 1
    x_padded = np.pad(x, ((padding, padding), (padding, padding)))
    output_height = (x_padded.shape[0] - effective_height) // stride + 1
    output_width = (x_padded.shape[1] - effective_width) // stride + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("有效卷积核不能大于填充后的输入")
    output = np.empty((output_height, output_width), dtype=np.float64)
    for output_row in range(output_height):
        for output_column in range(output_width):
            value = 0.0
            for kernel_row in range(kernel_height):
                for kernel_column in range(kernel_width):
                    input_row = output_row * stride + kernel_row * dilation
                    input_column = output_column * stride + kernel_column * dilation
                    value += x_padded[input_row, input_column] * kernel[kernel_row, kernel_column]
            output[output_row, output_column] = value
    return output

hand_x = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
hand_kernel = np.array([[1.0, 0.0], [0.0, -1.0]])
hand_correlation = cross_correlation_2d(hand_x, hand_kernel)
hand_convolution = cross_correlation_2d(hand_x, np.flip(hand_kernel, axis=(0, 1)))

assert hand_correlation.shape == (3, 3)
assert np.allclose(hand_correlation, -5.0)
assert np.allclose(hand_convolution, 5.0)
```

<div class="ml-stop">
  <strong>停下来推一推 1</strong>
  <p>不用代码，继续手算上例的 Y[1,2]。然后把核水平、垂直各翻转一次，说明为什么结果符号改变。最后判断：若核由训练学习，这个翻转是否减少可表示的函数集合？</p>
</div>

<MlQuiz kind="定义辨析" question="为什么深度学习框架用互相关却仍常命名为 Conv2d？" :options="['互相关与数学卷积对所有固定核数值相同', '学习型核可吸收翻转，二者表达能力相同，但固定核的数值方向仍可能不同', '框架会偷偷翻转输入', '只有一维信号才需要翻转']" :answer="1" explanation="核可学习时翻转只是参数重命名；固定 Sobel 等核仍必须核对定义。" />

<ConvolutionScanWorkbenchLab />

## 4. 输出几何：先算有效核，再算可落脚次数

一维方向上，输入长度 $H$、两侧填充 $P$、核尺寸 $K$、膨胀率 $D$、步幅 $S$。膨胀核在原输入上占据的有效宽度是

$$
K_{\text{eff}}=D(K-1)+1.
$$

填充后可供核起点移动的距离为 $H+2P-K_{\text{eff}}$。起点每次移动 $S$，所以输出长度为

$$
\boxed{
H_{\text{out}}
=\left\lfloor
\frac{H+2P-D(K-1)-1}{S}
\right\rfloor+1.
}
$$

宽度同理。向下取整不是记号装饰：最后不足一个完整步幅的位置不能放置核。若业务要求每个输入位置都有对应输出，必须明确使用哪种 `same` 约定；当 $S>1$ 或核为偶数时，左右填充可能不对称，不同框架的自动填充规则也可能不同。

```python
def convolution_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    effective_kernel = dilation * (kernel_size - 1) + 1
    numerator = input_size + 2 * padding - effective_kernel
    if numerator < 0:
        raise ValueError("有效卷积核大于填充后的输入")
    return numerator // stride + 1

assert convolution_output_size(32, 5, stride=2, padding=2) == 16
assert convolution_output_size(16, 3, stride=1, padding=1) == 16
assert convolution_output_size(16, 3, stride=1, padding=0, dilation=2) == 12
assert convolution_output_size(7, 3, stride=2, padding=1) == 4
```

### 4.1 `valid`、`same` 与边界语义

- `valid`：不补边界，只在核完整落入输入时计算，输出缩小；
- `same`：通常指选择填充使 $S=1$ 时输出与输入同尺寸；
- `full`：允许任何部分重叠，输出扩大，在普通 CNN 前向中较少直接使用。

零填充隐含“图像外部灰度为 0”。若输入已经做零均值，这可能尚可；若 0 是纯黑且训练图边缘总带黑框，网络可能学到边框捷径。反射、复制或循环填充表达不同的边界世界，应由物理过程决定。

<MlQuiz kind="几何手算" question="输入 32×32，5×5 核，膨胀率 1，填充 2，步幅 2，输出空间尺寸是多少？" :options="['14×14', '15×15', '16×16', '17×17']" :answer="2" explanation="floor((32+4−5)/2)+1=floor(31/2)+1=16；末端不足一步的部分被舍弃。" />

<MlQuiz kind="边界诊断" question="把零均值图像改成 0–255 原始灰度后继续使用零填充，最应优先检查什么？" :options="['边界是否出现与真实场景不一致的强黑框特征', '参数量是否随图像尺寸变化', 'Softmax 类别数是否自动增加', '卷积核是否停止共享']" :answer="0" explanation="填充值的物理含义随预处理改变；边界伪特征可能成为捷径。" />

## 5. 多通道卷积：先跨空间，再跨通道求和

设一个批次输入采用 `NCHW` 布局：

$$
X\in\mathbb R^{N\times C_{\text{in}}\times H\times W},
$$

权重为

$$
K\in\mathbb R^{C_{\text{out}}\times C_{\text{in}}\times K_h\times K_w}.
$$

输出为

$$
Y_{n,o,i,j}
=b_o+
\sum_{c=0}^{C_{\text{in}}-1}
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
K_{o,c,u,v}X_{n,c,iS_h+u-P_h,jS_w+v-P_w}.
$$

一个输出通道对应一个三维滤波器，它同时观察所有输入通道。RGB 的“红色竖边”可能要求红通道取正、绿蓝通道取负；第二层的通道已经不是颜色，而是上一层学到的特征。

```python
def conv2d_nchw(x, weight, bias=None, stride=1, padding=0):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("x 和 weight 必须分别是 NCHW 与 OIHW 四维张量")
    batch_size, input_channels, height, width = x.shape
    output_channels, weight_channels, kernel_height, kernel_width = weight.shape
    if input_channels != weight_channels:
        raise ValueError("输入通道数与卷积核通道数不一致")
    output_height = convolution_output_size(height, kernel_height, stride, padding)
    output_width = convolution_output_size(width, kernel_width, stride, padding)
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output = np.zeros((batch_size, output_channels, output_height, output_width))
    for sample in range(batch_size):
        for output_channel in range(output_channels):
            for output_row in range(output_height):
                for output_column in range(output_width):
                    patch = x_padded[
                        sample,
                        :,
                        output_row * stride:output_row * stride + kernel_height,
                        output_column * stride:output_column * stride + kernel_width,
                    ]
                    output[sample, output_channel, output_row, output_column] = np.sum(
                        patch * weight[output_channel]
                    )
                    if bias is not None:
                        output[sample, output_channel, output_row, output_column] += bias[output_channel]
    return output

multi_x = np.arange(2 * 2 * 4 * 4, dtype=np.float64).reshape(2, 2, 4, 4) / 10
multi_weight = np.ones((3, 2, 3, 3), dtype=np.float64)
multi_bias = np.array([0.0, 1.0, -1.0])
multi_output = conv2d_nchw(multi_x, multi_weight, multi_bias, padding=1)

assert multi_output.shape == (2, 3, 4, 4)
assert np.allclose(multi_output[:, 1] - multi_output[:, 0], 1.0)
assert np.allclose(multi_output[:, 2] - multi_output[:, 0], -1.0)
```

### 5.1 `NCHW` 与 `NHWC` 是接口合同

两种布局都合法，但不能凭 shape 猜测。例如 `(8, 32, 32, 32)` 在两种布局下都“看起来合理”。生产接口应记录：

- batch、channel、height、width 的轴序；
- dtype 与取值范围；
- RGB/BGR、灰度或多光谱通道语义；
- contiguous/stride 要求；
- 训练与推理预处理版本。

<MlQuiz kind="张量形状" question="NCHW 输入有 3 个通道，卷积核 OIHW 形状为 (16,3,5,5)。输出为什么有 16 个通道？" :options="['每个输出通道对应一组跨 3 个输入通道的 5×5 权重', '输入高恰好是 16', 'padding 自动复制通道', '每个输入像素都变成一个通道']" :answer="0" explanation="权重第一轴枚举输出滤波器；每个滤波器跨所有输入通道求和。" />

## 6. 让固定核先跑通案例，再把核交给学习

在训练 CNN 前，先用固定方向核检查数据和几何合同，能把“模型没学会”与“输入轴、灰度范围、方向或填充错了”分开。

对合成钢带小块，构造垂直 Sobel、水平 Sobel 和 Laplacian 核。每个核产生一张响应图，再取均值绝对响应与峰值作为诊断特征。

```python
vertical_sobel = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
horizontal_sobel = vertical_sobel.T
laplacian = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

diagnostic_features = np.empty((sample_count, 5), dtype=np.float64)
for sample_index in range(sample_count):
    image = images[sample_index, 0]
    vertical_map = cross_correlation_2d(image, vertical_sobel)
    horizontal_map = cross_correlation_2d(image, horizontal_sobel)
    laplacian_map = cross_correlation_2d(image, laplacian)
    diagnostic_features[sample_index] = [
        np.mean(np.abs(vertical_map)),
        np.max(np.abs(vertical_map)),
        np.mean(np.abs(horizontal_map)),
        np.max(np.abs(laplacian_map)),
        np.std(image),
    ]

class_feature_means = np.vstack([
    diagnostic_features[labels == label].mean(axis=0) for label in range(3)
])
assert class_feature_means.shape == (3, 5)
assert class_feature_means[1, 0] > class_feature_means[0, 0] * 4
assert class_feature_means[2, 3] > class_feature_means[0, 3] * 2.5
print("normal/scratch/pit 的诊断特征均值：\n", np.round(class_feature_means, 3))
```

固定核不是最终模型，却是很有价值的单元测试：竖划痕的垂直边缘响应应明显增强，点蚀的局部二阶响应应增强。若这两个关系在采集样本上完全不存在，应先检查相机方向、裁块和标签，而不是立即加深网络。

### 6.1 卷积层学习什么

训练时核不再是人工 Sobel。损失函数通过反向传播决定哪些局部模板有利于最终任务。第一层常能观察到方向、颜色或频率模式；更深层核位于抽象通道空间，不能简单当作一张 RGB 小图解释。

<MlQuiz kind="实验设计" question="固定 Sobel 核在上线前最适合承担什么职责？" :options="['替代全部监督学习', '作为输入方向、灰度和卷积定义的可解释烟雾测试', '证明 CNN 一定优于所有模型', '直接决定停线动作']" :answer="1" explanation="固定核能验证数据与算子合同，但不能替代任务训练、风险评估或控制逻辑。" />

## 7. 平移等变不是平移不变

令 $T_\Delta$ 把输入平移 $\Delta$。在无限网格或忽略边界时，步幅 1 的共享卷积满足

$$
\operatorname{Corr}(T_\Delta X,K)
=T_\Delta\operatorname{Corr}(X,K).
$$

输入划痕移动，响应图也移动，这叫**平移等变**（equivariance）。若最终分类不随位置变化，才叫**平移不变**（invariance）。两者关系可写成：

$$
\text{局部卷积产生等变特征}
\quad\longrightarrow\quad
\text{全局汇总产生近似不变决策}.
$$

全局平均池化就是一种汇总：

$$
g_c=\frac1{HW}\sum_{i,j}X_{cij}.
$$

但实际 CNN 并不严格平移等变，破坏来源包括边界填充、步幅、池化网格、裁剪、插值和数值实现。

```python
def shift_with_zeros(array, row_shift, column_shift):
    shifted = np.zeros_like(array)
    source_row_start = max(0, -row_shift)
    source_row_end = array.shape[0] - max(0, row_shift)
    source_column_start = max(0, -column_shift)
    source_column_end = array.shape[1] - max(0, column_shift)
    target_row_start = max(0, row_shift)
    target_column_start = max(0, column_shift)
    shifted[
        target_row_start:target_row_start + (source_row_end - source_row_start),
        target_column_start:target_column_start + (source_column_end - source_column_start),
    ] = array[source_row_start:source_row_end, source_column_start:source_column_end]
    return shifted

equivariance_x = rng.normal(size=(7, 7))
equivariance_kernel = rng.normal(size=(3, 3))
equivariance_y = cross_correlation_2d(equivariance_x, equivariance_kernel, padding=1)
shifted_x = shift_with_zeros(equivariance_x, 1, 1)
shifted_y = cross_correlation_2d(shifted_x, equivariance_kernel, padding=1)

# 右下平移后，排除因有限画幅裁掉的末端两行/列，内部等变关系精确成立。
equivariance_error = np.max(np.abs(shifted_y[1:-1, 1:-1] - equivariance_y[:-2, :-2]))
stride_two_original = cross_correlation_2d(equivariance_x, equivariance_kernel, stride=2, padding=1)
stride_two_shifted = cross_correlation_2d(shifted_x, equivariance_kernel, stride=2, padding=1)
stride_phase_error = np.linalg.norm(stride_two_original - stride_two_shifted)

assert equivariance_error < 1e-12
assert stride_phase_error > 1e-3
```

一步像素平移会改变步幅 2 的采样相位，因此输出不再只是平移。数据增强可以教模型容忍小位移，但不能把结构性采样误差“证明消失”。

<MlQuiz kind="概念边界" question="卷积层最直接提供哪种性质？" :options="['任意旋转不变', '输入平移时特征图相应平移的等变性', '对所有尺度不变', '分类输出绝对不受边界影响']" :answer="1" explanation="共享卷积在理想条件下平移等变；不变决策还需要汇总、训练和边界处理。" />

## 8. 感受野：深层单元究竟能看到多少原像素

某层单元在原输入上的**理论感受野**是所有可能影响它的输入坐标集合。用两个量递推：

- $r_l$：第 $l$ 层单元对应的感受野边长；
- $j_l$：第 $l$ 层相邻单元中心在原输入上的间距，也叫 jump。

初始 $r_0=j_0=1$。第 $l$ 层核、步幅、膨胀率分别为 $K_l,S_l,D_l$：

$$
j_l=j_{l-1}S_l,
$$

$$
\boxed{
r_l=r_{l-1}+(K_l-1)D_lj_{l-1}.
}
$$

三个步幅 1、膨胀率 1 的 $3\times3$ 卷积依次得到 $r=3,5,7$。它们与单个 $7\times7$ 核有相同理论范围，但中间插入两次非线性；若通道数相同，三个小核通常也比一个大核参数少。

```python
def receptive_field_schedule(layers):
    receptive_field = 1
    jump = 1
    ledger = []
    for kernel_size, stride, dilation in layers:
        receptive_field += (kernel_size - 1) * dilation * jump
        jump *= stride
        ledger.append((receptive_field, jump))
    return ledger

three_small = receptive_field_schedule([(3, 1, 1), (3, 1, 1), (3, 1, 1)])
downsampled = receptive_field_schedule([(3, 2, 1), (3, 1, 1)])
dilated = receptive_field_schedule([(3, 1, 1), (3, 1, 2), (3, 1, 4)])

assert three_small == [(3, 1), (5, 1), (7, 1)]
assert downsampled == [(3, 2), (7, 2)]
assert dilated[-1] == (15, 1)
```

### 8.1 理论感受野不等于有效感受野

理论感受野只说明“存在计算路径”。训练后，不同路径权重不同，真正对输出影响显著的区域常集中在中心，称为有效感受野。工程上可用遮挡、输入梯度、特征归因或受控平移实验估计，不能只看架构表。

### 8.2 感受野还要换算成物理尺度

若空间标定为 $0.08\text{ mm/pixel}$，$31\times31$ 感受野约覆盖 $2.48\times2.48$ mm。更换镜头或下采样后，像素尺度变化；同一网络结构看到的物理范围也变化。

<div class="ml-stop">
  <strong>停下来推一推 2</strong>
  <p>两层 3×3 卷积，第一层步幅 2，第二层步幅 1。逐层写出 r 与 j，再把最终 7×7 像素范围按 0.08 mm/pixel 换算为物理尺寸。为什么不能只说“第二层是 3×3”？</p>
</div>

<MlQuiz kind="递推手算" question="两层 3×3 卷积，第一层步幅 2、第二层步幅 1，理论感受野边长是多少？" :options="['5', '6', '7', '9']" :answer="2" explanation="第一层 r=3,j=2；第二层 r=3+(3−1)×2=7。" />

<ReceptiveFieldPlannerLab />

## 9. 步幅、池化与下采样：压缩前先处理混叠

### 9.1 最大池化

窗口 $\Omega_{ij}$ 内

$$
Y_{ij}=\max_{(u,v)\in\Omega_{ij}}X_{uv}.
$$

反向只把上游梯度发给前向最大值位置。若多个元素并列最大，框架的 tie-breaking 规则决定梯度给谁；这属于可复现性合同。

### 9.2 平均池化

$$
Y_{ij}=\frac1{|\Omega_{ij}|}\sum_{(u,v)\in\Omega_{ij}}X_{uv},
$$

反向把梯度平均分给窗口内元素。池化没有可学习权重，却仍有前向缓存、边界和反向规则。

```python
def pool2d_forward_backward(x, kernel_size=2, stride=2, mode="max", upstream=None):
    x = np.asarray(x, dtype=np.float64)
    output_height = (x.shape[0] - kernel_size) // stride + 1
    output_width = (x.shape[1] - kernel_size) // stride + 1
    output = np.empty((output_height, output_width))
    gradient = np.zeros_like(x)
    if upstream is None:
        upstream = np.ones((output_height, output_width))
    for output_row in range(output_height):
        for output_column in range(output_width):
            row_start = output_row * stride
            column_start = output_column * stride
            window = x[row_start:row_start + kernel_size, column_start:column_start + kernel_size]
            if mode == "max":
                flat_index = int(np.argmax(window))
                local_row, local_column = np.unravel_index(flat_index, window.shape)
                output[output_row, output_column] = window[local_row, local_column]
                gradient[row_start + local_row, column_start + local_column] += upstream[output_row, output_column]
            elif mode == "average":
                output[output_row, output_column] = window.mean()
                gradient[row_start:row_start + kernel_size, column_start:column_start + kernel_size] += (
                    upstream[output_row, output_column] / window.size
                )
            else:
                raise ValueError("mode 必须是 max 或 average")
    return output, gradient

pool_x = np.array([[1.0, 3.0, 2.0, 0.0], [4.0, 2.0, 1.0, 5.0],
                   [0.0, 2.0, 3.0, 1.0], [1.0, 6.0, 2.0, 4.0]])
max_output, max_gradient = pool2d_forward_backward(pool_x, mode="max")
average_output, average_gradient = pool2d_forward_backward(pool_x, mode="average")

assert np.allclose(max_output, [[4.0, 5.0], [6.0, 4.0]])
assert np.count_nonzero(max_gradient) == 4
assert np.allclose(average_gradient, 0.25)
```

### 9.3 下采样是采样，不只是“尺寸变小”

若输入含有接近 Nyquist 频率的细纹，直接每隔两个像素取样会发生混叠：不同高频纹理产生相同低分辨率结果。先低通再抽样可以减轻相位敏感性。

```python
checkerboard = (np.indices((8, 8)).sum(axis=0) % 2).astype(np.float64)
shifted_checkerboard = shift_with_zeros(checkerboard, 0, 1)
raw_downsample = checkerboard[::2, ::2]
raw_shifted_downsample = shifted_checkerboard[::2, ::2]

average_kernel = np.ones((2, 2), dtype=np.float64) / 4
blurred = cross_correlation_2d(checkerboard, average_kernel)
blurred_shifted = cross_correlation_2d(shifted_checkerboard, average_kernel)
blurred_downsample = blurred[::2, ::2]
blurred_shifted_downsample = blurred_shifted[::2, ::2]

raw_phase_gap = np.mean(np.abs(raw_downsample - raw_shifted_downsample))
blurred_phase_gap = np.mean(np.abs(blurred_downsample - blurred_shifted_downsample))
assert raw_phase_gap > 0.7
assert blurred_phase_gap < raw_phase_gap
```

对细划痕检测，过早的步幅 2 可能直接跨过一像素缺陷。可选策略包括延迟下采样、在下采样前低通、保留高分辨率分支或提高传感器空间分辨率；哪个有效必须按缺陷物理宽度验证。

<MlQuiz kind="采样诊断" question="一像素划痕在步幅 2 后时有时无，最可能的结构性原因是什么？" :options="['参数共享减少了类别数', '采样相位与混叠使高频小目标被跳过', '卷积参数量与图像尺寸无关', '偏置没有空间维度']" :answer="1" explanation="强下采样会对细小高频目标产生相位敏感；应检查低通与分辨率，而不是只调学习率。" />

## 10. 膨胀、1×1、深度可分离与转置卷积

### 10.1 膨胀卷积

膨胀率 $D$ 让相邻核元素隔 $D-1$ 个输入位置取样。$3\times3,D=2$ 的有效核为 $5\times5$，参数仍是 9 个。它能快速扩大感受野，但连续使用相同膨胀率可能形成栅格盲区。

### 10.2 1×1 点卷积

$1\times1$ 卷积不混合相邻位置，只在每个位置做通道线性变换：

$$
y_{oij}=b_o+\sum_cK_{oc}x_{cij}.
$$

它可调整通道数、形成瓶颈或在深度卷积后做通道混合。

### 10.3 深度可分离卷积

标准 $K\times K$ 卷积参数量为

$$
K^2C_{\text{in}}C_{\text{out}}.
$$

深度可分离卷积先对每个输入通道独立做空间卷积，再用 $1\times1$ 点卷积混合通道：

$$
K^2C_{\text{in}}+C_{\text{in}}C_{\text{out}}.
$$

```python
def separable_convolution_budget(kernel_size, input_channels, output_channels):
    standard = kernel_size ** 2 * input_channels * output_channels
    depthwise = kernel_size ** 2 * input_channels
    pointwise = input_channels * output_channels
    return standard, depthwise + pointwise

standard_weights, separable_weights = separable_convolution_budget(3, 64, 128)
assert standard_weights == 73_728
assert separable_weights == 8_768
assert separable_weights / standard_weights < 0.12
print("标准/深度可分离参数：", standard_weights, separable_weights)
```

参数减少不保证端到端延迟按同样比例下降。内存访问、算子融合、硬件并行度和 batch size 都会影响真实速度，必须在目标设备上基准测试。

### 10.4 转置卷积不是卷积的逆

把普通卷积写成线性算子 $y=Ax$，转置卷积计算 $A^Tz$。它常用于可学习上采样，也是普通卷积对输入反向的一部分；但通常不满足 $A^TA=I$，不能自动恢复被步幅、边界或核丢失的信息。“反卷积”作为其别名容易误导，真正的 deconvolution 是另一个逆问题。

<MlQuiz kind="算子边界" question="转置卷积最准确的线性代数解释是什么？" :options="['总能精确求出原输入', '应用卷积线性算子的转置 Aᵀ，而不是一般意义的逆 A⁻¹', '把核参数全部取倒数', '只是一种最大池化']" :answer="1" explanation="转置不等于逆；下采样与卷积通常已丢失信息。" />

## 11. 卷积反向传播：共享意味着累加

重新考虑单通道、步幅 1、无填充：

$$
Y_{ij}=\sum_{u,v}K_{uv}X_{i+u,j+v}.
$$

设上游梯度

$$
G_{ij}=\frac{\partial L}{\partial Y_{ij}}.
$$

### 11.1 对核求梯度

同一个 $K_{uv}$ 在每个输出位置使用。总导数必须把所有路径相加：

$$
\boxed{
\frac{\partial L}{\partial K_{uv}}
=\sum_{i,j}G_{ij}X_{i+u,j+v}.
}
$$

这本身又是输入与上游梯度之间的互相关。

### 11.2 对输入求梯度

一个输入像素 $X_{pq}$ 被所有覆盖它的窗口使用：

$$
\boxed{
\frac{\partial L}{\partial X_{pq}}
=\sum_{u,v}G_{p-u,q-v}K_{uv},
}
$$

越界 $G$ 视为 0。这里索引出现反向偏移，所以常说“输入梯度像用翻转核做 full convolution”。边界像素被窗口覆盖次数较少，其梯度统计天然不同于内部像素。

### 11.3 对偏置求梯度

每个输出通道共享一个偏置：

$$
\frac{\partial L}{\partial b_o}
=\sum_{n,i,j}G_{n,o,i,j}.
$$

若损失对 batch 取 mean，$G$ 已经包含相应缩放，不要在 `db` 再除一次 batch size。

<ConvolutionBackwardCollectorLab />

<MlQuiz kind="梯度审计" question="为什么一个卷积核元素的梯度必须对所有输出位置求和？" :options="['为了让输出变成方形', '同一参数沿多个空间使用路径影响损失，总导数要累加每条路径', '因为输入一定归一化', '只有偏置需要共享']" :answer="1" explanation="参数共享在前向重复使用一个变量；反向的分支规则要求收集全部贡献。" />

## 12. 从零实现通用单通道前向与反向

加入步幅、填充和膨胀后，最可靠的实现方式不是背诵翻转规则，而是回到索引：前向每次读了哪个 `x_padded[input_row,input_column]`，反向就把对应贡献加回同一坐标。

```python
def conv2d_forward_backward(x, kernel, upstream, stride=1, padding=0, dilation=1):
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    upstream = np.asarray(upstream, dtype=np.float64)
    output = cross_correlation_2d(x, kernel, stride, padding, dilation)
    if upstream.shape != output.shape:
        raise ValueError(f"上游梯度应为 {output.shape}，实际为 {upstream.shape}")

    x_padded = np.pad(x, ((padding, padding), (padding, padding)))
    gradient_x_padded = np.zeros_like(x_padded)
    gradient_kernel = np.zeros_like(kernel)

    for output_row in range(output.shape[0]):
        for output_column in range(output.shape[1]):
            gradient_output = upstream[output_row, output_column]
            for kernel_row in range(kernel.shape[0]):
                for kernel_column in range(kernel.shape[1]):
                    input_row = output_row * stride + kernel_row * dilation
                    input_column = output_column * stride + kernel_column * dilation
                    gradient_kernel[kernel_row, kernel_column] += (
                        gradient_output * x_padded[input_row, input_column]
                    )
                    gradient_x_padded[input_row, input_column] += (
                        gradient_output * kernel[kernel_row, kernel_column]
                    )

    if padding == 0:
        gradient_x = gradient_x_padded
    else:
        gradient_x = gradient_x_padded[padding:-padding, padding:-padding]
    return output, gradient_x, gradient_kernel

gradient_x_input = rng.normal(size=(5, 6))
gradient_kernel_input = rng.normal(size=(3, 2))
gradient_output_upstream = rng.normal(size=(3, 4))
gradient_output, gradient_x_result, gradient_kernel_result = conv2d_forward_backward(
    gradient_x_input,
    gradient_kernel_input,
    gradient_output_upstream,
    stride=2,
    padding=1,
    dilation=1,
)

assert gradient_output.shape == gradient_output_upstream.shape == (3, 4)
assert gradient_x_result.shape == gradient_x_input.shape
assert gradient_kernel_result.shape == gradient_kernel_input.shape
assert np.isfinite(gradient_x_result).all()
assert np.isfinite(gradient_kernel_result).all()
```

实现中的 `+=` 是核心语义，不是编码风格。若改成 `=`：

- 核梯度只保留最后一个使用位置；
- 输入梯度只保留最后一个覆盖窗口；
- shape 仍正确，loss 也可能下降，但导数已经错了。

<div class="ml-stop">
  <strong>停下来推一推 3</strong>
  <p>对 3×3 输入、2×2 核、2×2 上游梯度，列出 K[0,0] 的四条贡献。再列出中心输入 X[1,1] 被哪些输出窗口覆盖。把代码中的 += 改成 = 会各自保留哪一项？</p>
</div>

<MlQuiz kind="实现定位" question="卷积反向的 dK shape 正确但有限差分失败，最优先检查哪一处？" :options="['输出文件名', '共享参数贡献是否用 += 跨所有 batch 与空间位置累加', '图片是否为正方形', '类别名称是否按字母排序']" :answer="1" explanation="漏加共享路径是卷积反向最典型的静默错误；shape 不会暴露它。" />

## 数值验证

解析推导必须接受独立数值审计。对标量损失

$$
L(X,K)=\sum_{i,j}Y_{ij}G_{ij},
$$

中心差分为

$$
\frac{\partial L}{\partial\theta}
\approx
\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}.
$$

下面逐坐标检查核与输入。使用 float64、固定小张量和 $\epsilon=10^{-6}$，避免把大模型随机训练当作梯度证明。

```python
def convolution_scalar_loss(x, kernel, upstream, stride=1, padding=0, dilation=1):
    output = cross_correlation_2d(x, kernel, stride, padding, dilation)
    return float(np.sum(output * upstream))

epsilon = 1e-6
numeric_gradient_kernel = np.zeros_like(gradient_kernel_input)
for index in np.ndindex(gradient_kernel_input.shape):
    kernel_plus = gradient_kernel_input.copy()
    kernel_minus = gradient_kernel_input.copy()
    kernel_plus[index] += epsilon
    kernel_minus[index] -= epsilon
    numeric_gradient_kernel[index] = (
        convolution_scalar_loss(gradient_x_input, kernel_plus, gradient_output_upstream, stride=2, padding=1)
        - convolution_scalar_loss(gradient_x_input, kernel_minus, gradient_output_upstream, stride=2, padding=1)
    ) / (2 * epsilon)

numeric_gradient_x = np.zeros_like(gradient_x_input)
for index in np.ndindex(gradient_x_input.shape):
    x_plus = gradient_x_input.copy()
    x_minus = gradient_x_input.copy()
    x_plus[index] += epsilon
    x_minus[index] -= epsilon
    numeric_gradient_x[index] = (
        convolution_scalar_loss(x_plus, gradient_kernel_input, gradient_output_upstream, stride=2, padding=1)
        - convolution_scalar_loss(x_minus, gradient_kernel_input, gradient_output_upstream, stride=2, padding=1)
    ) / (2 * epsilon)

kernel_relative_error = np.max(np.abs(numeric_gradient_kernel - gradient_kernel_result)) / max(
    1.0, np.max(np.abs(numeric_gradient_kernel)), np.max(np.abs(gradient_kernel_result))
)
x_relative_error = np.max(np.abs(numeric_gradient_x - gradient_x_result)) / max(
    1.0, np.max(np.abs(numeric_gradient_x)), np.max(np.abs(gradient_x_result))
)

assert kernel_relative_error < 1e-8
assert x_relative_error < 1e-8
print("dK/dX 最大相对误差：", kernel_relative_error, x_relative_error)
```

### 13.1 `im2col`：卷积也可以写成矩阵乘法

把每个局部窗口展平成一行，得到补丁矩阵

$$
X_{\text{col}}\in\mathbb R^{H_{\text{out}}W_{\text{out}}\times K_hK_w}.
$$

核展平为向量 $k$，则

$$
\operatorname{vec}(Y)=X_{\text{col}}k.
$$

这揭示了卷积的稀疏、权重绑定结构，也便于复用高性能矩阵乘法；显式 `im2col` 会复制重叠窗口，真实库常使用隐式展开或专用内核。

```python
def im2col_single(x, kernel_height, kernel_width, stride=1, padding=0):
    x_padded = np.pad(x, ((padding, padding), (padding, padding)))
    output_height = (x_padded.shape[0] - kernel_height) // stride + 1
    output_width = (x_padded.shape[1] - kernel_width) // stride + 1
    rows = []
    for output_row in range(output_height):
        for output_column in range(output_width):
            patch = x_padded[
                output_row * stride:output_row * stride + kernel_height,
                output_column * stride:output_column * stride + kernel_width,
            ]
            rows.append(patch.reshape(-1))
    return np.vstack(rows), output_height, output_width

matrix_x = rng.normal(size=(6, 7))
matrix_kernel = rng.normal(size=(3, 3))
patch_matrix, matrix_height, matrix_width = im2col_single(matrix_x, 3, 3, stride=2, padding=1)
matrix_output = (patch_matrix @ matrix_kernel.reshape(-1)).reshape(matrix_height, matrix_width)
loop_output = cross_correlation_2d(matrix_x, matrix_kernel, stride=2, padding=1)

assert patch_matrix.shape == (12, 9)
assert np.allclose(matrix_output, loop_output, atol=1e-12)
```

<MlQuiz kind="数值审计" question="为什么不能仅凭训练 loss 下降断言卷积反向正确？" :options="['loss 下降已经等价于逐坐标导数相等', '错误梯度仍可能偶然下降；有限差分用独立前向扰动检查局部导数', '只有分类任务需要梯度', 'NumPy 不支持卷积']" :answer="1" explanation="优化表现不是导数证明；小张量 float64 差分能直接暴露漏加、索引和翻转错误。" />

### 13.2 可选 PyTorch 对照

仓库标准验证环境不为一段对照新增大型框架依赖。若已有 PyTorch，可使用完全相同的 float64 输入、核、步幅、填充、上游梯度和 reduction 比较前向、`x.grad` 与 `weight.grad`。

```python
# 不执行：仅在另行准备的 PyTorch 环境中运行。
import torch

x_torch = torch.tensor(gradient_x_input[None, None], dtype=torch.float64, requires_grad=True)
weight_torch = torch.tensor(gradient_kernel_input[None, None], dtype=torch.float64, requires_grad=True)
upstream_torch = torch.tensor(gradient_output_upstream[None, None], dtype=torch.float64)

output_torch = torch.nn.functional.conv2d(x_torch, weight_torch, stride=2, padding=1)
loss_torch = (output_torch * upstream_torch).sum()
loss_torch.backward()

assert np.allclose(output_torch.detach().numpy()[0, 0], gradient_output, atol=1e-12)
assert np.allclose(x_torch.grad.numpy()[0, 0], gradient_x_result, atol=1e-12)
assert np.allclose(weight_torch.grad.numpy()[0, 0], gradient_kernel_result, atol=1e-12)
```

## 14. 用卷积特征完成一次案例基线

卷积响应图本身还不是分类器。先用训练卷的诊断特征均值和方差标准化，再以训练类中心做最近中心分类，建立一个可解释基线。它不是完整 CNN，却把“局部核—特征图—空间汇总—分类”主链完整跑通。

```python
feature_mean = diagnostic_features[train_mask].mean(axis=0)
feature_scale = diagnostic_features[train_mask].std(axis=0, ddof=0)
standardized_features = (diagnostic_features - feature_mean) / feature_scale

class_centers = np.vstack([
    standardized_features[train_mask & (labels == label)].mean(axis=0)
    for label in range(3)
])

def nearest_center_predict(features, centers):
    squared_distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(squared_distances, axis=1)

validation_prediction = nearest_center_predict(standardized_features[validation_mask], class_centers)
test_prediction = nearest_center_predict(standardized_features[test_mask], class_centers)
validation_accuracy = np.mean(validation_prediction == labels[validation_mask])
test_accuracy = np.mean(test_prediction == labels[test_mask])

test_confusion = np.zeros((3, 3), dtype=np.int64)
for truth, prediction in zip(labels[test_mask], test_prediction):
    test_confusion[truth, prediction] += 1

assert validation_accuracy > 0.95
assert test_accuracy > 0.95
assert test_confusion.sum() == 144
print("验证/测试准确率：", validation_accuracy, test_accuracy)
print("测试混淆矩阵：\n", test_confusion)
```

这个结果只证明合成规则容易被固定核区分。它不证明真实钢带达到同样性能，更不证明可自动放行。真正训练 CNN 时，应把固定核换成可学习参数，并冻结：

- 训练目标和类别代价；
- 模型选择只看验证卷；
- 测试卷只做最终一次评估；
- 缺陷级与钢卷级指标；
- 置信区间、拒绝选项和漂移分层；
- 推理预处理、模型权重、阈值和软件版本。

<MlQuiz kind="案例解释" question="合成测试准确率接近 100%，最合理的结论是什么？" :options="['真实产线可立即自动放行', '实现与合成数据规则一致；真实性能、标签质量和行动风险仍未被证明', 'CNN 对所有图像任务都完美', '不再需要分组测试']" :answer="1" explanation="合成实验是算法与实现检查，不是外部有效性证据。" />

## 复杂度与工程边界

### 15. 计算、内存与内核选择

对 `NCHW` 标准卷积，忽略 bias，乘加次数约为

$$
N H_{\text{out}}W_{\text{out}}C_{\text{out}}C_{\text{in}}K_hK_w.
$$

参数内存只是工程成本的一部分。训练还要保存输入或补丁、输出激活、梯度、优化器状态与临时 workspace。

```python
def convolution_system_estimate(
    batch_size, input_channels, output_channels,
    output_height, output_width, kernel_size, bytes_per_value=4,
):
    macs = (
        batch_size * output_height * output_width
        * output_channels * input_channels * kernel_size ** 2
    )
    parameter_values = output_channels * input_channels * kernel_size ** 2 + output_channels
    output_values = batch_size * output_channels * output_height * output_width
    parameter_bytes = parameter_values * bytes_per_value
    output_bytes = output_values * bytes_per_value
    return macs, parameter_bytes, output_bytes

system_macs, system_parameter_bytes, system_output_bytes = convolution_system_estimate(
    batch_size=8,
    input_channels=64,
    output_channels=128,
    output_height=128,
    output_width=128,
    kernel_size=3,
)
assert system_macs == 9_663_676_416
assert system_parameter_bytes == 295_424
assert system_output_bytes == 67_108_864
print("MAC / 参数 MiB / 单层输出 MiB：", system_macs, system_parameter_bytes / 2**20, system_output_bytes / 2**20)
```

这个例子只有约 0.28 MiB 参数，单层输出却有 64 MiB，乘加接近 97 亿。高分辨率检测中，“参数很少”不能推导出“训练显存很少”或“推理很快”。

### 15.1 算法实现不是固定答案

常见实现路线包括：

- 直接循环：定义清楚，适合教学与极小核；
- `im2col + GEMM`：复用矩阵乘法，但可能复制补丁；
- implicit GEMM：不显式物化完整补丁矩阵；
- Winograd：可减少特定小核乘法，但数值与适用尺寸有限；
- FFT 卷积：大核可能有利，小核常被变换开销抵消；
- 厂商专用内核：依据 shape、dtype、硬件和 workspace 动态选择。

性能验收必须报告硬件、软件版本、warm-up、batch、dtype、布局、同步方式、峰值显存和分位延迟，不能只贴一次计时。

### 15.2 混合精度与可复现性

float16/bfloat16 可提高吞吐并降低激活内存，但要检查：

- 累加是否使用更高精度；
- 小梯度下溢与 loss scaling；
- 算法选择是否确定性；
- 不同硬件和库版本的容差；
- 量化后小缺陷召回是否分层下降。

### 15.3 稠密预测不能过早压缩空间

分类可用全局池化得到不变表示；分割、定位和尺寸测量需要保留坐标对应。若一路压成 $1\times1$，缺失的边界与小目标信息不能仅靠转置卷积凭空恢复。编码器—解码器通常用跳接把高分辨率特征送到解码端，但这仍需验证对实际最小缺陷的覆盖。

<MlQuiz kind="系统预算" question="某层卷积参数只有 0.28 MiB，是否可据此断言训练显存很小？" :options="['可以，显存只存参数', '不可以；高分辨率激活、梯度、优化器状态和 workspace 可能远大于参数', '可以，只要核是 3×3', '只有 CPU 才保存激活']" :answer="1" explanation="训练显存由整条计算图共同决定，高分辨率特征图常是主要部分。" />

## 16. 失效模式与定位顺序

| 现象 | 优先检查 | 典型根因 | 验证与处置 |
|---|---|---|---|
| 输出尺寸差 1 | $K_{eff}$、floor、左右填充 | 把 dilation 或非对称 same 漏掉 | 手算一维，再断言 shape |
| 固定 Sobel 方向相反 | 运算定义、相机轴 | 卷积/互相关混淆、图像转置 | 小矩阵逐项乘积核对 |
| 边缘误报集中 | 填充值与裁块 | 黑框捷径、反射边界不物理 | 按距边缘距离分层评估 |
| 划痕移动 1 px 就漏检 | stride/池化相位 | 混叠、小目标被跳过 | 受控平移扫描、先低通 |
| 训练高、随机测试高、换卷骤降 | 分组键 | 同卷重叠 patch 泄漏 | 按 `coil_id` 重切 |
| dK shape 对但差分失败 | 共享路径 | `=` 覆盖 `+=`、漏 batch 维 | 小张量逐坐标差分 |
| dX 边界错误 | padding crop | 裁切索引错、翻转规则硬背错 | 按前向读地址散射累加 |
| 深层看不到完整缺陷 | 感受野与物理尺度 | 网络过浅、标定变化 | 递推 $r,j$ 并换算 mm |
| dilation 出现栅格盲区 | 采样点覆盖 | 多层固定膨胀率 | 混合 dilation 或普通卷积 |
| 参数少但延迟高 | 目标硬件 profile | 内存带宽、算子碎片、低利用率 | 真机 benchmark 与融合 |
| 概率很高却错得集中 | 分层与校准 | 新材质、曝光漂移、未知缺陷 | OOD/拒绝选项与人工复核 |
| 转置卷积出现棋盘格 | kernel/stride 覆盖不均 | 上采样重叠模式 | resize-conv 或覆盖均匀设计 |

推荐定位顺序：

1. 固定一个 $4\times4$ 小输入，手算前向和输出 shape；
2. 用固定 Sobel 核确认轴方向、通道、灰度与 padding；
3. 用 float64 小张量对 `dX,dK,db` 做中心差分；
4. 再与第二个独立实现或框架对照；
5. 受控平移、边界、缩放和细目标测试结构性质；
6. 按钢卷、材质、宽度、相机和距边缘距离分层评估；
7. 最后才增加深度、通道数或更换架构。

## 17. 本章可执行检查清单

### 数据与任务

- [ ] 空间标定、相机方向、灰度链路和裁块协议已版本化。
- [ ] 相关 patch 按钢卷或缺陷事件分组切分。
- [ ] 训练统计量没有读取验证或测试卷。
- [ ] 分类概率与放行、复核、停线动作分离。

### 前向与几何

- [ ] 能区分互相关与数学卷积，并手算小矩阵。
- [ ] 先算 $K_{eff}$，再算输出尺寸并断言 shape。
- [ ] `NCHW/NHWC`、通道语义、dtype 和范围明确。
- [ ] 参数量、MAC 和激活内存分别核算。

### 结构性质

- [ ] 能区分平移等变与平移不变。
- [ ] 对边界、一步平移和步幅相位做受控测试。
- [ ] 逐层递推理论感受野并换算物理尺度。
- [ ] 下采样前评估混叠和最小缺陷尺寸。

### 反向与数值

- [ ] `dK` 汇总 batch 和所有空间使用位置。
- [ ] `dX` 汇总所有覆盖窗口，并正确裁掉 padding。
- [ ] `db` 的 reduction 与 loss mean/sum 合同一致。
- [ ] float64 小张量的前向、`dX`、`dK` 通过有限差分。

### 部署与监控

- [ ] 在目标硬件记录吞吐、P50/P95/P99、峰值显存与 warm-up。
- [ ] 模型、预处理、阈值、相机配置和校准版本绑定。
- [ ] 按材质、相机、位置和缺陷尺寸监控召回与误报。
- [ ] 未知缺陷、漂移和低置信样本进入可追溯复核流程。

## 18. 练习与最小实现

### 练习 1：手算二维前向与反向

给定

$$
X=
\begin{bmatrix}
1&2&0\\
3&1&2\\
0&1&4
\end{bmatrix},\quad
K=
\begin{bmatrix}
1&-1\\
0&2
\end{bmatrix},\quad
G=
\begin{bmatrix}
1&2\\
-1&1
\end{bmatrix}.
$$

手算互相关输出 $Y$，再计算全部 $dK$ 和 $dX$。明确每个核参数的四次使用，以及中心输入收集的覆盖路径。

### 练习 2：补全 NumPy 卷积

扩展本章实现，支持：

- `NCHW` batch；
- 多输入与多输出通道；
- 非方形 kernel/stride/padding/dilation；
- bias；
- `dX,dK,db`；
- 至少 20 个随机坐标的 float64 有限差分；
- 无效 shape 和通道不匹配拒绝。

### 练习 3：受控平移与混叠报告

构造宽度为 1、2、3、4 像素的竖线，逐像素平移穿过输入。比较：

1. 步幅 1 卷积；
2. 步幅 2 卷积；
3. 最大池化；
4. 低通后步幅 2。

画出峰值响应随位置变化的曲线，并说明哪种配置满足钢带最小缺陷验收要求。

### 练习 4：小型 CNN 案例

在独立训练环境中建立 `Conv-ReLU-Conv-ReLU-GlobalAveragePool-Linear`：

- 严格按 `coil_id` 切分；
- 只用训练卷估计标准化；
- 报告类别混淆、宏平均召回和卷级 bootstrap 区间；
- 与本章固定核最近中心基线比较；
- 做边界、一步平移、曝光偏移和未知缺陷压力测试。

```python
# TODO：实现支持 NCHW、多通道、非方形超参数的 Conv2d forward/backward。
#
# 验收要求：
# 1. 采用 OIHW 权重布局并严格校验 shape；
# 2. 支持 stride、padding、dilation 与 bias；
# 3. dK 跨 batch 和空间位置累加，dX 正确裁掉 padding；
# 4. 用 float64 中心差分抽查 dX、dK、db；
# 5. 与已有框架在相同输入、权重、dtype、reduction 下对照。
```

## 19. 本章结论

1. CNN 用局部连接编码局部性，用参数共享编码空间复用。
2. 深度学习库的 `Conv2d` 通常计算不翻核的互相关；学习型核可吸收翻转。
3. 输出几何由有效核、填充、步幅和向下取整共同决定。
4. 卷积参数量与输入空间尺寸无关，计算量和激活内存并非如此。
5. 多通道卷积对空间窗口和输入通道共同求和，每个输出通道有一组滤波器。
6. 理想步幅 1 卷积平移等变，不自动保证最终预测平移不变。
7. 感受野递推必须同时维护范围 $r$ 与跳距 $j$，并换算到物理尺度。
8. 步幅和池化是采样操作；小目标检测必须显式审计混叠与相位。
9. 膨胀、1×1、深度可分离和转置卷积分别解决覆盖、通道混合、效率和上采样问题，不能混为一谈。
10. 共享核梯度必须累加所有使用位置，输入梯度必须累加所有覆盖窗口。
11. 小张量手算、有限差分、第二实现对照和受控变换共同构成卷积验收门禁。
12. 合成高准确率只证明实现符合合成规则，不证明真实产线性能或自动放行安全。

## 来源与许可

卷积层、二维互相关、卷积的稀疏与权重绑定矩阵、padding、stride、多输入/输出通道、1×1 卷积、池化、平移等变与全局平均池化的知识边界，参考 Kevin P. Murphy 的 *Probabilistic Machine Learning: An Introduction* 源 PDF 第 498–506 页（§14.1–14.2.3）。膨胀卷积、转置卷积与深度可分离卷积参考同书源 PDF 第 516–519 页（§14.4.1–14.4.3）。

从全连接视觉模型的参数问题到共享局部探测器、padding、stride、下采样、训练、多通道卷积与池化的讲解线索，交叉参考 CMU 10-301/10-601 Lecture 17 *MAP, CNN, RNN* 源 PDF 第 31–89 页。局部连接、共享权重、通道参数量、池化与步幅卷积参考 Jonathan Richard Shewchuk 的 Berkeley CS189/289A *Machine Learning* 讲义源 PDF 第 115–119 页（Lecture 19）。

本章没有复用上述资料的原图、课堂案例或措辞。冷轧钢带 864 个小块、36 卷分组、缺陷生成规则、固定核基线、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作。教学合成数据不代表真实钢带缺陷机理、相机性能、质量标准、复核阈值、放行规范或停线策略；PyTorch 对照明确标为可选未执行内容，仓库标准验证环境未为此新增大型依赖。

<MlLessonProgress
  lesson-id="15"
  next-href="/courses/machine-learning/16-rnn"
  next-label="第 16 章：循环神经网络（RNN）"
/>
