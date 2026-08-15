---
course: deep-learning
edition: industrial
title: 初始化、归一化与稳定训练
description: 从层间信号方差推导 Xavier 与 He 初始化，比较 BatchNorm、LayerNorm 和 RMSNorm 的统计接口，解释残差路径、梯度裁剪、FP16/BF16 与动态 loss scaling。
outline: [2, 3]
---

<DlLessonHero number="02" module="现代训练" title="初始化、归一化与稳定训练" question="网络没有报错、loss 也能计算，为什么有效梯度仍可能在几十层之后消失、爆炸或被数值格式抹掉？" duration="建议 260–340 分钟" :outcomes="['用方差传播推导 Xavier 与 He 初始化的适用假设', '区分 BatchNorm、LayerNorm、RMSNorm 的归一化轴与训练/推理状态', '解释 pre-norm 残差路径为什么更利于深层梯度传播', '诊断 FP16 underflow/overflow，并正确安排 loss scaling、unscale 与裁剪']" />

“稳定训练”不是 loss 没有 NaN。一个模型可能数值有限，却长期处在激活饱和、梯度过小、更新相对参数尺度失衡或训练/推理统计不一致的状态。本章把稳定性拆成四条可以观测的信号：前向激活尺度、反向梯度尺度、残差主干的可达性和低精度数值范围。

## 贯穿案例：把 18 层分类器扩展到 96 层

视觉团队已经有一个能训练的 18 层网络，现在希望在相同数据上扩到 96 层。宽度保持 512，激活使用 GELU，训练格式为 BF16。加深后出现三种现象：

1. 无残差网络的末层激活标准差降到 $10^{-4}$；
2. post-norm 残差网络能训练，但前 20 层梯度明显小于后部；
3. 切到 FP16 后偶发 overflow，降低学习率仍不稳定。

这三个问题分别指向信号传播、残差/归一化拓扑和数值格式。把它们统一归因于“学习率太大”只会延迟定位。

每次训练至少记录以下逐层摘要：

```python
from collections import defaultdict

signals = defaultdict(list)

def record_signal(name):
    def hook(module, inputs, output):
        tensor = output.detach().float()
        signals[name].append({
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "max_abs": tensor.abs().max().item(),
            "finite": bool(tensor.isfinite().all()),
        })
    return hook
```

Hook 会增加同步和内存开销，只在诊断运行中打开；正式吞吐基准必须关闭或单独报告。

<DlQuiz kind="诊断边界" question="训练 loss 有限且下降，能否证明各层信号稳定？" :options="['可以，有限 loss 足以证明一切', '不可以，局部梯度可能极小、部分通道可能饱和，最终 loss 仍可下降', '只有 CNN 需要检查逐层信号', '只需检查最后一层']" :answer="1" explanation="标量 loss 会压缩大量内部状态。稳定性需要逐层激活、梯度、更新比和有限值证据。" />

## 1. 从方差传播推导初始化

考虑宽度为 $n$ 的线性层：

$$
z_j=\sum_{i=1}^{n}w_{ji}x_i.
$$

在 $w_{ji}$ 与 $x_i$ 独立、均值为 0、不同项近似不相关的假设下：

$$
\operatorname{Var}(z_j)
=n\operatorname{Var}(w_{ji})\operatorname{Var}(x_i).
$$

若希望线性层前后方差近似不变，需要

$$
\operatorname{Var}(w_{ji})\approx\frac1n.
$$

这就是 fan-in 尺度的来源。它不是保证真实网络方差恒定的定理，因为独立、零均值和相同分布会被训练、卷积共享、注意力和残差连接破坏；它提供的是合理起点。

### 1.1 Xavier 初始化平衡前向与反向

若同时考虑 fan-in 与 fan-out，Xavier/Glorot 初始化常使用

$$
\operatorname{Var}(W)=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}.
$$

它适合近似线性或对称激活。对于 sigmoid/tanh，权重过大会迅速把输入推入饱和区；过小则层层缩小信号。

### 1.2 He 初始化补偿 ReLU 门控

若 $z$ 关于 0 对称，ReLU 大约把一半元素置零，因此二阶矩近似减半：

$$
\mathbb E[\operatorname{ReLU}(z)^2]\approx\frac12\mathbb E[z^2].
$$

为了补偿，He 初始化令

$$
\operatorname{Var}(W)=\frac{2}{n_{\mathrm{in}}}.
$$

对于 GELU、SiLU 或门控 MLP，精确因子不同。工程上通常从成熟实现的 gain 开始，再用逐层统计验证，而不是为每个激活背一个常数。

```python
import numpy as np

rng = np.random.default_rng(7)
x = rng.normal(size=(4096, 512))

for variance, label in [(1 / 512, "Xavier"), (2 / 512, "He")]:
    w = rng.normal(scale=np.sqrt(variance), size=(512, 512))
    z = x @ w
    relu = np.maximum(z, 0)
    print(label, round(z.std(), 3), round(relu.std(), 3))
```

<SignalPropagationLab />

<DlQuiz kind="推导检查" question="对 ReLU 网络仍使用 Var(W)=1/fan_in，深度增加时最可能发生什么？" :options="['ReLU 丢掉约一半二阶矩，信号可能逐层衰减', '激活一定逐层翻倍', '初始化只影响第一层', '方差与激活无关']" :answer="0" explanation="在线性层保持方差后，ReLU 门控继续降低二阶矩；He 初始化用约 2/fan_in 补偿这一效应。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>卷积层的 fan-in 应怎样计算？写出输入通道数、核高、核宽的关系，并说明 groups 改变后为什么要重新计算。</div>

## 2. 激活函数改变的不只是“非线性强弱”

| 激活 | 正区间 | 负区间 | 主要稳定性问题 |
|---|---|---|---|
| ReLU | 线性 | 0 | dead unit、均值偏正，但导数简单 |
| GELU | 平滑门控 | 保留小负值 | 计算更复杂，常用于 Transformer |
| SiLU/Swish | $x\sigma(x)$ | 平滑负尾 | 非单调小区间，常与门控 MLP 配合 |
| tanh | 有界 | 有界 | 大绝对值输入进入饱和，梯度接近 0 |

选择激活要连同初始化、归一化、残差拓扑和硬件 kernel 一起评估。某个激活在单算子 benchmark 更慢，不等于端到端更慢；融合实现可能改变结论。

### 2.1 监控相对更新量

参数更新的绝对值没有跨层可比性。更有用的是

$$
r_l=\frac{\lVert\Delta\theta_l\rVert_2}
{\lVert\theta_l\rVert_2+\epsilon}.
$$

若某些层的 $r_l$ 长期比其他层小几个数量级，它们可能几乎冻结；若持续过大，则参数每步都在重写。相对更新量也必须结合归一化不变性解释，不能设一个跨模型通用阈值。

## 3. 归一化层到底对哪些轴统计

设序列张量形状为 $[B,T,D]$。

### 3.1 BatchNorm

BatchNorm 对同一通道收集 batch（以及卷积中的空间位置）统计：

$$
\widehat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad
y=\gamma\widehat x+\beta.
$$

训练时使用当前 batch 统计并更新 running statistics；推理时使用累计统计。小 batch、非独立样本、分布漂移或忘记切换 `eval()` 都会造成训练/推理错位。

### 3.2 LayerNorm

LayerNorm 对每个 token 的特征维 $D$ 计算均值和方差，不依赖其他样本：

$$
\operatorname{LN}(x)=\gamma\odot
\frac{x-\mu_D}{\sqrt{\sigma_D^2+\epsilon}}+\beta.
$$

它适合变长序列和小 batch。代价是把均值与尺度都移除，且其统计语义与 BatchNorm 不同。

### 3.3 RMSNorm

RMSNorm 不减均值，只按均方根缩放：

$$
\operatorname{RMSNorm}(x)=
\gamma\odot\frac{x}{\sqrt{\frac1D\sum_{j=1}^{D}x_j^2+\epsilon}}.
$$

它减少了一项中心化操作，广泛用于现代 LLM。不能从“公式更短”直接推断所有硬件上都更快；融合 kernel、shape 和内存访问才决定端到端时间。

<DlQuiz kind="形状判断" question="对形状 [B,T,D] 的语言模型隐藏状态，LayerNorm 通常在哪个轴上计算统计量？" :options="['跨整个 B×T×D', '对每个 token 的 D 维特征', '只对 batch 轴 B', '只对时间轴 T']" :answer="1" explanation="标准 LayerNorm 对每个样本位置的最后若干特征维归一化，不混合不同 token 或不同 batch 样本。" />

## 4. 残差路径是深层网络的信号高速路

残差块写成

$$
x_{l+1}=x_l+F_l(x_l).
$$

其 Jacobian 为

$$
\frac{\partial x_{l+1}}{\partial x_l}=I+J_{F_l}.
$$

即使残差分支的 Jacobian 很小，恒等项 $I$ 仍提供直接梯度路径。这不保证任意深度都稳定，但解释了残差结构为何比纯堆叠更容易优化。

### 4.1 Post-norm 与 pre-norm

Post-norm：

$$
x_{l+1}=\operatorname{Norm}(x_l+F_l(x_l)).
$$

Pre-norm：

$$
x_{l+1}=x_l+F_l(\operatorname{Norm}(x_l)).
$$

Pre-norm 把恒等路径留在归一化之外，深层梯度通常更容易通过，因此现代大模型常采用 pre-norm 或其变体。代价是残差流尺度可能随深度增长，需要初始化缩放、残差缩放或额外归一化设计。

<div class="dl-stop"><strong>停下来推一推 2</strong>分别对 pre-norm 和 post-norm 写出从 $x_{l+1}$ 到 $x_l$ 的梯度路径。哪一条含有不经过归一化 Jacobian 的恒等项？</div>

## 5. 梯度裁剪限制一次异常，不修复长期错误

全局范数裁剪计算

$$
G=\sqrt{\sum_l\lVert g_l\rVert_2^2},
$$

当 $G>c$ 时统一缩放

$$
g_l\leftarrow g_l\frac{c}{G}.
$$

统一缩放保持不同参数梯度之间的方向关系。逐元素 clip 会改变方向，通常只在明确需要时使用。

裁剪适合处理偶发尖峰，不能修复持续过高学习率、错误 loss reduction、坏数据或 overflow。日志必须同时记录裁剪前范数、阈值和被裁剪比例；若 90% step 都触发裁剪，阈值已经成为隐式学习率控制器。

```python
import numpy as np

def clip_by_global_norm(gradients, max_norm, eps=1e-12):
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    total = np.sqrt(sum(float(np.sum(g * g)) for g in gradients))
    scale = min(1.0, max_norm / (total + eps))
    clipped = [g * scale for g in gradients]
    clipped_total = np.sqrt(sum(float(np.sum(g * g)) for g in clipped))
    assert clipped_total <= max_norm + 1e-9
    return clipped, total, scale
```

<DlQuiz kind="故障诊断" question="几乎每一步 global norm 都从 200 被裁到 1。最合理的下一步是什么？" :options="['宣布训练稳定，因为最终范数等于 1', '检查学习率、loss reduction、数据尖峰和数值溢出，裁剪可能正在掩盖根因', '继续把阈值降到 0', '删除梯度日志']" :answer="1" explanation="持续裁剪说明原始信号长期异常或尺度合同错误。裁剪保证单步有界，却不能证明优化过程健康。" />

## 6. FP16、BF16 与 FP32 的真正差别

浮点格式由符号、指数和尾数组成。指数位决定动态范围，尾数位决定有效精度。

| 格式 | 指数位 | 尾数位 | 典型特点 |
|---|---:|---:|---|
| FP32 | 8 | 23 | 范围与精度较高，内存和带宽成本大 |
| FP16 | 5 | 10 | 尾数比 BF16 多，但范围窄，易 overflow/underflow |
| BF16 | 8 | 7 | 范围接近 FP32，精度较低，通常更易训练大模型 |

混合精度不是“所有张量改成 FP16”。常见策略是低精度做矩阵乘和存储激活，归约、部分 loss、optimizer state 或 master weight 保持 FP32。具体策略由框架、硬件和算子实现决定。

### 6.1 Loss scaling

若真实梯度太小，在 FP16 中可能下溢为 0。先把 loss 乘 $S$：

$$
\widetilde L=SL,qquad
\nabla\widetilde L=S\nabla L,
$$

backward 后再除以 $S$，理论上恢复原梯度。若 $S$ 太大，中间梯度会 overflow。动态 loss scaler 在无 overflow 时逐步增大 $S$，检测到非有限值时跳过更新并减小 $S$。

正确顺序是：backward 得到 scaled gradient → unscale → 检查非有限值 → 裁剪 → optimizer step。对 scaled gradient 直接裁剪会把阈值也隐式放大。

<MixedPrecisionLab />

<DlQuiz kind="顺序判断" question="使用 FP16 loss scaling 和 global norm clipping 时，应先做什么？" :options="['先对 scaled gradient 裁剪，再 unscale', '先 unscale，再检查非有限值并裁剪', '只需修改权重衰减', '每一步重置 loss scale 为 1']" :answer="1" explanation="裁剪阈值定义在真实梯度尺度上。必须先 unscale，否则阈值含义会随 loss scale 改变。" />

## 7. 激活检查点交换计算与内存

反向传播通常需要前向激活。Activation checkpointing 只保存部分边界，反向时重新计算中间激活。它降低激活内存，代价是额外计算和更复杂的随机性处理。

Checkpoint 区域若包含 dropout 等随机算子，重计算必须恢复一致 RNG 状态，否则反向对应的函数与原前向不同。还要注意带副作用或状态更新的层；例如把 BatchNorm running statistics 在重计算中更新两次会破坏语义。

## 8. 稳定训练的最小审计面板

| 信号 | 聚合方式 | 告警只说明什么 |
|---|---|---|
| 激活 mean/std/max | 逐层、按时间抽样 | 表示尺度异常，不直接给根因 |
| 梯度 global norm | clip 前后 | 单步梯度尺度和裁剪频率 |
| update/weight ratio | 逐参数组 | 有效更新相对大小 |
| finite ratio | 激活、梯度、状态 | 首个非有限位置 |
| loss scale | 每 step | 数值范围是否频繁退让 |
| overflow skip | 计数和连续长度 | 实际少做了多少优化步 |
| throughput/memory | 固定 shape | 稳定性措施的系统代价 |

发布门禁：能定位首个非有限张量；逐层信号在至少两个随机种子中可重放；checkpoint 恢复后的下一步与连续训练一致；训练与 `eval()` 的归一化状态经过单独测试。

<div class="dl-stop"><strong>停下来推一推 3</strong>模型从 FP32 改成 BF16 后吞吐提高 40%，验证损失略差。写出至少六项核对：有效 batch、loss scale、归约格式、优化器状态、随机种子、训练 token 之外还缺什么？</div>

## 9. 推荐论文阅读

<DlPaperCard title="Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" meta="Ioffe & Szegedy · 2015" priority="必读" question="算法中哪些统计量来自当前 mini-batch，哪些状态在推理时使用？" reading="先读 Algorithm 1 与训练/推理说明，再看卷积版本。把论文的原始解释与后续实践分开：本课程使用它的算法定义，不把 internal covariate shift 当作唯一机制。" href="https://arxiv.org/pdf/1502.03167" />

<DlPaperCard title="Root Mean Square Layer Normalization" meta="Zhang & Sennrich · 2019" priority="选读" question="去掉 re-centering 后保留了什么不变性，又放弃了什么？" reading="重点读第 2 节公式和复杂度讨论；实验表用于理解适用范围，不把小幅平均提升外推到所有 LLM。" href="https://arxiv.org/pdf/1910.07467" />

<DlPaperCard title="Deep Residual Learning for Image Recognition" meta="He et al. · 2015" priority="跨章必读" question="残差参数化解决的是表示能力不足，还是优化退化？" reading="先读引言中的 degradation problem，再看残差块公式与 identity shortcut 消融。第 04 章会进入完整架构演进。" href="https://arxiv.org/pdf/1512.03385" />

## 10. 练习

1. 在独立零均值假设下推导线性层前向方差，并写出反向梯度方差的对应表达式。
2. 用 NumPy 堆叠 50 层 ReLU 网络，比较小方差、Xavier 与 He 初始化的激活 std。
3. 对一个 $[B,T,D]$ 张量手算 LayerNorm；再改变另一个 batch 样本，验证当前 token 输出不变。
4. 构造 FP16 小梯度，比较无 loss scaling、固定 scaling 和过大 scaling 的 underflow/overflow。
5. 实现 global norm clipping，验证所有梯度张量使用同一个缩放系数，且裁剪后范数不超过阈值。
6. 设计 BatchNorm 恢复测试：训练模式更新 running statistics，保存后恢复并在 `eval()` 下比较输出。

## 本章小结

- Xavier 与 He 初始化来自方差传播假设；真实网络必须用逐层统计复核。
- BatchNorm、LayerNorm、RMSNorm 的统计轴和状态不同，不能只按名称替换。
- 残差恒等路径为深层梯度提供直达通道，pre-norm 进一步避免恒等路径穿过归一化 Jacobian。
- 梯度裁剪限制异常单步，不证明根因消失。
- FP16 与 BF16 的范围和精度不同；loss scaling 必须在 unscale 后再检查与裁剪。
- 激活检查点以重计算换内存，随机算子与状态更新必须保持语义一致。

<DlLessonProgress lesson-id="02" next-href="/courses/deep-learning/03-transfer-diagnostics" next-label="下一章：迁移与诊断" />
