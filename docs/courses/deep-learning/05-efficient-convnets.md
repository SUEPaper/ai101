---
course: deep-learning
edition: industrial
title: MobileNet、EfficientNet 与 ConvNeXt
description: 从深度可分离卷积、倒残差、复合缩放与现代卷积训练配方出发，区分理论 FLOPs、内存访问和真实设备延迟。
outline: [2, 3]
---

<DlLessonHero number="05" module="计算机视觉" title="MobileNet、EfficientNet 与 ConvNeXt" question="模型的 FLOPs 降了十倍，为什么手机、边缘 GPU 或浏览器上的延迟可能只降两倍，甚至更慢？" duration="建议 250–330 分钟" :outcomes="['推导 depthwise separable convolution 的参数和 MAC 账本', '解释 MobileNetV2 的 inverted residual 与 linear bottleneck', '审计 EfficientNet 复合缩放的适用边界', '用真实设备测量比较 ConvNeXt 与高效网络，而不是把 FLOPs 当延迟']" />

第 04 章把准确率提升放进结构和训练系统。本章加入第三个约束：目标硬件。高效模型不是“小 ResNet”的同义词；它同时设计算术强度、激活尺寸、算子形状、量化路径和编译器支持。

## 贯穿案例：边缘相机上的零件分类

设备有 4 核 ARM CPU、共享内存 GPU 和 2 GiB 内存，batch 固定为 1。要求 P95 低于 18 ms、峰值工作集低于 350 MiB、断网运行。候选模型的桌面 GPU benchmark 没有决策价值，必须在目标 runtime、线程、温度和功耗模式下复测。

| 账本 | 单位 | 常见误读 |
|---|---|---|
| 参数 | 个数/bytes | 参数少就一定快 |
| MAC/FLOPs | 次数 | 理论计算少就等于低延迟 |
| 激活峰值 | bytes | 只统计层输出，忽略临时 workspace |
| 内存访问 | bytes/s | 忽略 depthwise 的低复用 |
| kernel 数 | launches | 忽略许多小算子的调度开销 |
| 端到端延迟 | P50/P95 ms | 只测预热后的单个算子 |

<DlQuiz kind="部署判断" question="两个模型 FLOPs 相同，为什么真实延迟仍可能不同？" :options="['不可能，FLOPs 唯一决定延迟', '算子形状、内存访问、融合、并行度和 runtime 支持都可能不同', '只因为参数随机种子不同', '只有分类任务才会不同']" :answer="1" explanation="FLOPs 没有编码数据搬运、kernel launch、硬件利用率和编译器路径。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>写一份 batch=1 的延迟测量协议：包括预热、同步、线程、输入预处理、重复次数、分位数、温度与正确性检查。</div>

## 1. 深度可分离卷积拆开两个混合

标准卷积同时做空间混合和通道混合，参数量为

$$
K^2C_{in}C_{out}.
$$

depthwise convolution 对每个输入通道独立做 $K\times K$，参数 $K^2C_{in}$；pointwise $1\times1$ 再做通道混合，参数 $C_{in}C_{out}$。比例为

$$
\frac{K^2C_{in}+C_{in}C_{out}}{K^2C_{in}C_{out}}
=\frac1{C_{out}}+\frac1{K^2}.
$$

当 $K=3$、$C_{out}$ 很大时，理论成本接近标准卷积的九分之一。

```python
def conv_cost(cin, cout, kernel, height, width):
    standard = height * width * kernel**2 * cin * cout
    separable = height * width * (kernel**2 * cin + cin * cout)
    return standard, separable

standard, separable = conv_cost(64, 128, 3, 56, 56)
assert standard / separable > 8
assert separable == 56 * 56 * (9 * 64 + 64 * 128)
```

<DepthwiseConvLab />

<DlQuiz kind="算子账本" question="depthwise convolution 本身会充分混合不同输入通道吗？" :options="['会，每个输出都读取全部输入通道', '不会，每个通道独立做空间卷积，跨通道混合通常交给 pointwise 卷积', '只在 7×7 核时会', '只在训练时会']" :answer="1" explanation="depthwise 的 groups 等于输入通道数，因此通道间没有直接连接。" />

### 1.1 理论省算为什么可能不快

标准 GEMM/卷积往往能高复用权重和激活。depthwise 每个通道工作很小，可能受内存带宽、向量宽度和 launch overhead 限制。若 runtime 把 depthwise、BN 和 activation 融合，情况会改善；若退回通用实现，理论优势可能丢失。

## 2. MobileNetV2：在高维空间做非线性

MobileNetV1 主要使用 depthwise separable convolution。MobileNetV2 引入 inverted residual：先用 $1\times1$ 把低维输入扩张到高维，在高维做 depthwise 空间混合，再线性投影回低维。若 shape 与 stride 允许，shortcut 连接两端的窄表示。

为什么最后一层不加 ReLU？低维瓶颈中的逐坐标截断更容易永久删除信息，因此使用 linear bottleneck。结构顺序是

$$
x\xrightarrow{1\times1\ \mathrm{expand}}z
\xrightarrow{K\times K\ \mathrm{depthwise}}u
\xrightarrow{1\times1\ \mathrm{linear}}y,
\qquad y=x+\mathrm{proj}(u).
$$

```python
def inverted_residual_shapes(batch, height, width, cin, expansion, cout, stride):
    hidden = cin * expansion
    return {
        "expand": (batch, height, width, hidden),
        "depthwise": (batch, height // stride, width // stride, hidden),
        "project": (batch, height // stride, width // stride, cout),
        "shortcut": stride == 1 and cin == cout,
    }

shapes = inverted_residual_shapes(1, 56, 56, 24, 6, 24, 1)
assert shapes["expand"] == (1, 56, 56, 144)
assert shapes["shortcut"]
```

<DlQuiz kind="块结构" question="为什么称为 inverted residual？" :options="['shortcut 连接宽表示，内部是窄表示', 'shortcut 连接窄瓶颈，块内部先扩张到更宽表示，与经典 bottleneck 的宽窄关系相反', '因为反向传播被倒置', '因为卷积核旋转了 180°']" :answer="1" explanation="经典 ResNet bottleneck 常在两端宽、中间窄；MobileNetV2 两端窄、内部扩张。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>扩张倍率从 6 降到 2 会同时改变哪些量：pointwise 参数、depthwise 参数、激活峰值、表达容量和可能的硬件利用率？</div>

## 3. MobileNetV3：架构搜索也要服从设备

MobileNetV3 把硬件感知搜索、NetAdapt、SE 模块和 h-swish 组合起来。核心教训不是记住某个 block 表，而是搜索目标必须包含真实设备延迟。若代理设备或 runtime 不同，找到的“最优”结构可能失效。

量化也会改变选择。某些激活和 SE 路径在 INT8 上需要额外 scale 或退回高精度；因此要在量化后模型上测延迟和精度，而非先按 FP32 选型再假设量化收益相同。

## 4. EfficientNet：同时缩放深、宽、分辨率

只加深会增加串行长度，只加宽会快速增加 $C^2$ 计算，只增分辨率会扩大每层空间成本。EfficientNet 从一个搜索得到的基线出发，使用

$$
d=\alpha^\phi,\qquad w=\beta^\phi,\qquad r=\gamma^\phi,
$$

并约束 $\alpha\beta^2\gamma^2\approx2$，使每增加一档 $\phi$，理论计算约翻倍。系数是经验搜索结果，不是跨任务常数；目标尺寸、训练数据和硬件都可能要求不同配比。

<CompoundScalingLab />

<DlQuiz kind="缩放判断" question="宽度扩大 2 倍、其他不变时，中间标准卷积的主要计算量近似怎样变化？" :options="['2 倍', '4 倍', '8 倍', '不变']" :answer="1" explanation="若输入与输出通道都扩大 2 倍，Cin×Cout 增至 4 倍。" />

## 5. ConvNeXt：重新审计“卷积已经落后”

ConvNeXt 以 ResNet 为起点，逐项吸收现代训练与结构选择：更大的 depthwise kernel、inverted bottleneck、较少激活与归一化、LayerNorm、独立下采样层，以及更强训练配方。它证明部分“Transformer 优势”曾与训练 recipe、宏观 stage 比例和算子选择混杂。

这不意味着 ConvNeXt 在所有设备最快。$7\times7$ depthwise、channels-last、LayerNorm 和大通道 pointwise 对不同 runtime 的映射差异很大。公平比较需要相同输入、精度、编译选项和真实 batch。

```python
def arithmetic_intensity(macs, bytes_read, bytes_written):
    moved = bytes_read + bytes_written
    if moved <= 0:
        raise ValueError("moved bytes must be positive")
    return macs / moved

standard_ai = arithmetic_intensity(90_000_000, 4_000_000, 2_000_000)
depthwise_ai = arithmetic_intensity(9_000_000, 3_500_000, 2_000_000)
assert standard_ai > depthwise_ai
```

<DlQuiz kind="证据辨析" question="ConvNeXt 的主要研究意义是什么？" :options="['证明注意力永远无效', '在现代训练和结构条件下重新建立强卷积基线，分离架构类别与训练配方的影响', '删除所有卷积', '只追求最少参数']" :answer="1" explanation="它是一次受控现代化实验，而不是“卷积永久战胜 Transformer”的结论。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>同一设备上模型 A 为 1 GFLOP/12 ms，模型 B 为 2 GFLOPs/9 ms。列出至少三种解释，并设计 profiler 证据区分计算受限、带宽受限和 launch 受限。</div>

## 6. 设备级选型协议

1. 固定输入张量、预处理和 batch；
2. 分别测 FP32、FP16/BF16、INT8 的精度与速度；
3. 预热并显式同步，报告 P50/P95 与样本数；
4. 记录 runtime、编译器、线程、功耗和温度；
5. 捕获峰值内存和 kernel 时间线；
6. 用目标数据切片报告准确率，不只看 ImageNet；
7. 导出后比较 logits、top-k 与异常输入行为。

选择结果要画 Pareto 前沿：任何同时更慢、更大、更差的点被支配；剩余点再按业务约束选择。一个模型可能在服务器吞吐上优胜，在手机 batch=1 上落后。

## 7. 推荐论文阅读

<DlPaperCard title="MobileNetV2: Inverted Residuals and Linear Bottlenecks" meta="Sandler et al. · 2018" priority="必读 · 移动端核心" question="为何非线性放在扩张空间，shortcut 又连接窄瓶颈？" reading="精读 manifold 直觉、block 表与 ablation；自己写出每步 shape 和 MAC。" href="https://arxiv.org/pdf/1801.04381" />

<DlPaperCard title="EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" meta="Tan & Le · 2019" priority="必读 · 缩放" question="复合系数如何把深度、宽度、分辨率放在同一资源约束下？" reading="读 scaling 方法、B0 搜索边界和迁移结果；区分搜索基线与缩放规则。" href="https://arxiv.org/pdf/1905.11946" />

<DlPaperCard title="A ConvNet for the 2020s" meta="Liu et al. · 2022" priority="必读 · 公平基线" question="哪些 ViT 时代的结构与训练选择能迁回卷积网络，哪些收益不能归因于单一部件？" reading="按论文的逐步 modernize 顺序读消融表，再检查下游检测与分割而非只看分类。" href="https://arxiv.org/pdf/2201.03545" />

## 8. 练习与失效门禁

1. 为标准卷积与 depthwise separable convolution 生成逐层 MAC 和激活表。
2. 在目标设备比较 fused 与 unfused block，记录 kernel 数、内存带宽与端到端延迟。
3. 画 MobileNetV2 一个 stride=2 block 的 shape 图，说明为何没有 identity shortcut。
4. 在固定计算倍率下设计三组深/宽/分辨率配比，按小目标召回比较。
5. 对一个 ConvNeXt 与一个 ViT 做相同训练 recipe、分辨率和部署 runtime 的对照。
6. 构造量化校准集，检查总体准确率之外的少数类与暗光切片。

## 本章小结

- depthwise separable convolution 把空间与通道混合拆开，大幅降低理论 MAC，但可能受内存与 kernel 效率限制。
- inverted residual 在高维做非线性、低维保持线性瓶颈，shortcut 连接窄表示。
- EfficientNet 的复合系数是一种受约束缩放方法，不是跨任务固定配方。
- ConvNeXt 说明训练 recipe 与宏观设计会混入架构比较，卷积基线必须现代化。
- 真实设备的 P95 延迟、峰值内存、量化精度和正确性才是部署证据。

<DlLessonProgress lesson-id="05" next-href="/courses/deep-learning/06-two-stage-detection" next-label="下一章：两阶段目标检测" />
