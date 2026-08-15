---
course: deep-learning
edition: industrial
title: AlexNet、VGG、GoogLeNet 与 ResNet
description: 不重复卷积基础，从 ImageNet 时代四类架构的预算分配、感受野、分支与残差路径出发，建立骨干网络的结构审计方法。
outline: [2, 3]
---

<DlLessonHero number="04" module="计算机视觉" title="AlexNet、VGG、GoogLeNet 与 ResNet" question="当卷积、池化和反向传播已经会用，经典架构真正改变的是哪一条计算、优化与证据接口？" duration="建议 260–340 分钟" :outcomes="['按参数、FLOPs、激活和感受野比较经典视觉骨干', '解释 AlexNet 的训练系统、VGG 的规则堆叠、Inception 的分支预算和 ResNet 的优化路径', '用残差块 Jacobian 说明恒等捷径的作用边界', '为工业分类基线建立结构、预训练与部署门禁']" />

本章假设已经理解二维卷积、步幅、padding、池化和卷积反向传播，不再从滤波器滑窗讲起。我们关心的是 2012–2015 年四次结构变化：AlexNet 证明规模化卷积系统能在 GPU 上训练；VGG 用规则小核堆叠换取深度；GoogLeNet 用分支和 $1\times1$ 降维管理预算；ResNet 用恒等捷径改写深度网络的优化路径。

## 贯穿案例：高分辨率铸件缺陷分类

输入是 $2048\times2048$ 灰度铸件图，缺陷只占几十像素。最终设备要求单图 P95 延迟低于 30 ms，召回率按缺陷尺寸切片报告。选择骨干不能只看 ImageNet top-1；还要回答：下采样前是否保留微小缺陷、预训练输入统计是否兼容、激活峰值是否可承受、部署后算子是否高效。

| 架构 | 主要结构动作 | 原始论文中的系统条件 | 今天仍值得保留的思想 |
|---|---|---|---|
| AlexNet | 大核首层、ReLU、重叠池化、Dropout | 双 GPU、强数据增强 | 结果来自训练系统整体 |
| VGG-16 | 重复 $3\times3$ 卷积 | 高参数量与计算量 | 规则 stage 接口便于复用 |
| GoogLeNet | 多尺度分支、$1\times1$ 瓶颈 | 控制计算预算 | 宽度与尺度可显式分配 |
| ResNet-50 | bottleneck residual block | BN 与残差联合 | 恒等路径改善可优化性 |

<DlQuiz kind="结构审计" question="把输入从 224×224 改为 448×448，卷积骨干的主要计算量近似怎样变化？" :options="['不变', '约 2 倍', '约 4 倍', '约 8 倍']" :answer="2" explanation="在通道与层数不变时，空间位置数随边长平方增长；448 是 224 的两倍，因此主要卷积 FLOPs 约增四倍。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>同一 2048 图像，整图缩放到 224 与切成 16 个 512 crop 再分类，分别丢失什么信息？计算预算、缺陷像素尺寸和图像级标签应怎样变化？</div>

## 1. 先统一四本账

卷积层参数量为 $K_hK_wC_{in}C_{out}$，乘加量近似为

$$
H_{out}W_{out}K_hK_wC_{in}C_{out}.
$$

参数决定权重存储，FLOPs/MACs 近似计算，激活决定训练显存和数据搬运，感受野描述单个输出可依赖的输入范围。四者不是一个指标。下面的代码把 stage 合同写成可检查账本：

```python
def conv_record(h, w, cin, cout, kernel=3, stride=1):
    hout, wout = h // stride, w // stride
    params = kernel * kernel * cin * cout
    macs = hout * wout * params
    activation = hout * wout * cout
    return hout, wout, params, macs, activation

h, w, params, macs, activation = conv_record(224, 224, 64, 128, 3, 2)
assert (h, w) == (112, 112)
assert params == 3 * 3 * 64 * 128
assert macs == activation * 3 * 3 * 64
```

<ArchitectureBudgetLab />

## 2. AlexNet：突破来自系统组合

AlexNet 的第一层使用 $11\times11$、stride 4，快速降低空间尺寸。今天看起来激进，但它受当时 GPU 内存与吞吐约束。更重要的不是背层数，而是四项组合：ReLU 改善优化速度；随机裁剪和翻转扩大有效数据；Dropout 约束大规模全连接层；双 GPU 切分让模型可以训练。

论文还使用 local response normalization，后来通常被 BatchNorm 等替代。历史模型中的每个部件不都应原样继承。复现时应区分“对结论必要的机制”和“被时代硬件绑定的实现”。

<DlQuiz kind="历史证据" question="为什么不能把 AlexNet 的胜利只总结成“大卷积核更好”？" :options="['因为论文没有卷积', '因为结果同时依赖 ReLU、增强、Dropout、GPU 规模和数据，且大核并未成为后续默认', '因为只使用 CPU', '因为准确率没有提升']" :answer="1" explanation="架构、优化、数据和硬件共同构成实验系统；孤立一个部件会错读证据。" />

### 2.1 早期下采样的风险

若 32 像素缺陷在输入缩放后只剩 3–4 像素，大步幅首层可能在产生有用表示前混叠。可选修正包括保留更高输入、减小 stem stride、使用多 crop 或专门检测器。每一种都必须重新测量 FLOPs、激活与真实缺陷召回。

## 3. VGG：规则性换来昂贵预算

两个 $3\times3$、stride 1 卷积的理论感受野等于一个 $5\times5$，三个约等于 $7\times7$，但中间多了非线性。忽略 bias，若输入输出通道都是 $C$：

$$
2\cdot 3^2C^2=18C^2 < 25C^2=5^2C^2.
$$

不过这不意味着 VGG 便宜。通道逐 stage 增长，VGG-16 约 138M 参数，分类头尤其昂贵。它的长期影响是规则接口：若 stage 形状稳定，检测与分割可以提取多尺度特征。

```python
def receptive_field(kernels, strides):
    rf, jump = 1, 1
    records = []
    for kernel, stride in zip(kernels, strides):
        rf += (kernel - 1) * jump
        jump *= stride
        records.append((rf, jump))
    return records

records = receptive_field([3, 3, 2, 3, 3], [1, 1, 2, 1, 1])
assert records[-1] == (14, 2)
```

<DlQuiz kind="感受野" question="连续两个 3×3、stride 1 卷积的理论感受野是多少？" :options="['3×3', '4×4', '5×5', '9×9']" :answer="2" explanation="第二层的每个位置覆盖第一层相邻三个位置，每个第一层位置又覆盖输入 3×3，合并后跨度为 5。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>理论感受野覆盖整个零件并不等于模型真的使用整个零件。怎样用遮挡、梯度或受控背景替换判断有效感受野与背景依赖？</div>

## 4. GoogLeNet：把尺度选择放进分支

Inception 模块并行计算 $1\times1$、$3\times3$、$5\times5$ 和池化分支，再沿通道拼接。若直接在高维输入上做大核，预算会爆炸，因此先用 $1\times1$ 降维。例如从 192 通道生成 32 个 $5\times5$ 通道：直接卷积参数为 $192\cdot32\cdot25=153600$；先降到 16 通道则为 $192\cdot16+16\cdot32\cdot25=15872$。

分支并不自动代表“模型懂多尺度”。需要检查各分支通道预算、拼接后的带宽、真实目标尺寸和导出引擎对并行小算子的效率。辅助分类头曾帮助深层网络传梯度，但残差和现代训练后未必必要。

<DlQuiz kind="预算判断" question="Inception 中大核分支前的 1×1 卷积主要起什么作用？" :options="['改变图像分辨率为 1×1', '在空间尺寸不变时压缩或混合通道，降低后续大核成本', '删除所有非线性', '等价于全局池化']" :answer="1" explanation="1×1 卷积在每个空间位置做通道投影，常用于瓶颈降维。" />

## 5. ResNet：学习残差而不是强迫每层重写表示

普通块学习 $H(x)$；残差块写成

$$
y=x+F(x;W).
$$

若最优映射接近恒等，残差分支只需学习小修正。反向传播时

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x},
$$

因此存在不经过残差权重连乘的恒等项。对多块展开还能看成许多不同深度路径的组合。它解释了优化便利，却不保证任意深度都稳定：激活位置、归一化、投影 shortcut、残差尺度和初始化仍关键。

<ResidualPathLab />

### 5.1 Basic block 与 bottleneck

ResNet-18/34 常用两个 $3\times3$；ResNet-50/101 常用 $1\times1$ 压缩、$3\times3$ 空间混合、$1\times1$ 扩张。若通道或分辨率改变，shortcut 需要投影。加法要求两路 shape 完全一致：

```python
import numpy as np

x = np.zeros((2, 56, 56, 256), dtype=np.float32)
residual = np.ones_like(x) * 0.05
assert x.shape == residual.shape
y = x + residual
assert y.shape == (2, 56, 56, 256)
assert np.allclose(y.mean(), 0.05)
```

投影 shortcut 不再是严格恒等，但仍提供短路径。若把每个 shortcut 都换成重变换，就削弱了结构目的。

<DlQuiz kind="残差接口" question="残差块两路 shape 不一致时，最合理的处理是什么？" :options="['依赖广播自动相加', '用明确的投影和 stride 对齐，并审计它是否只在 stage 边界出现', '删除 shortcut', '把所有值展平成向量']" :answer="1" explanation="残差相加需要相同 shape。投影是显式结构选择，不能依赖广播掩盖错误。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>若训练 152 层 ResNet 仍发散，列出至少四个不能被“它有 shortcut”排除的原因，并为每个原因指定一个观测量。</div>

## 6. 从论文模型到工业骨干

迁移时先建立三个基线：原始预训练分辨率的线性探测；保持 stem 的全量微调；面向小缺陷调整 stem/输入的结构版本。比较同一实体切分、增强、预算与评价切片。若改 stem，应登记预训练权重未加载或插值的层，并重新做学习率分组。

部署门禁至少包括：

- 输入 resize/crop 与训练完全一致；
- 灰度到三通道的映射有版本；
- 参数量、MAC、峰值显存、P50/P95 延迟同时报告；
- 按缺陷像素面积切片召回；
- 预训练许可与权重来源可追溯；
- 导出后 logits 与参考实现误差有阈值。

## 7. 推荐论文阅读

<DlPaperCard title="ImageNet Classification with Deep Convolutional Neural Networks" meta="Krizhevsky, Sutskever & Hinton · 2012" priority="必读 · 系统起点" question="哪些结果来自结构，哪些来自数据增强、正则化、GPU 与训练规模的组合？" reading="读架构图、训练细节和消融；特别记录今天已经不再沿用的 LRN 与双 GPU 切分。" href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf" />

<DlPaperCard title="Very Deep Convolutional Networks for Large-Scale Image Recognition" meta="Simonyan & Zisserman · 2014" priority="必读 · 结构对照" question="规则 3×3 堆叠如何把深度变成受控实验变量，代价又集中在哪里？" reading="先读 configurations 表，再计算每个 stage 的参数、激活与感受野；不要只记 VGG-16 名称。" href="https://arxiv.org/pdf/1409.1556" />

<DlPaperCard title="Deep Residual Learning for Image Recognition" meta="He et al. · 2015" priority="必读 · 核心" question="degradation problem 与过拟合有什么区别，残差参数化提供了什么优化证据？" reading="精读 plain/residual 对照、shortcut 公式和深度实验；把训练误差与验证误差分开看。" href="https://arxiv.org/pdf/1512.03385" />

## 8. 练习与发布门禁

1. 为 AlexNet、VGG-16、GoogLeNet、ResNet-50 生成逐 stage shape/参数/MAC 表。
2. 计算输入从 224 提到 384 时 FLOPs 近似倍率，并用 profiler 验证端到端倍率为何不同。
3. 在同一小型数据集比较 plain 20 层与 residual 20 层的训练误差；不要只看测试分数。
4. 为小缺陷任务设计 stem stride 消融，报告缺陷尺寸切片和峰值激活。
5. 导出一个残差模型，逐层比较参考与部署引擎的最大绝对误差。

## 本章小结

- 经典架构演进是预算、训练系统和优化路径的共同变化，不是模型名年表。
- 参数、FLOPs、激活和感受野分别回答不同问题；分辨率改变会重写账本。
- VGG 的规则堆叠便于分析但昂贵；Inception 用分支与瓶颈分配多尺度预算。
- ResNet 的恒等项提供短梯度路径，但不能替代初始化、归一化和训练诊断。
- 工业骨干必须用目标尺寸切片、真实延迟和导出正确性决定，而不是只看 ImageNet top-1。

<DlLessonProgress lesson-id="04" next-href="/courses/deep-learning/05-efficient-convnets" next-label="下一章：高效卷积网络" />
