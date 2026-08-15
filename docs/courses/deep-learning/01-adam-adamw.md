---
course: deep-learning
edition: industrial
title: Adam、AdamW 与学习率调度
description: 从梯度时间序列推导 Adam 的矩估计、偏差修正和逐坐标步长，解释 AdamW 为什么要解耦权重衰减，并用固定训练预算审计学习率调度。
outline: [2, 3]
---

<DlLessonHero number="01" module="现代训练" title="Adam、AdamW 与学习率调度" question="优化器看到的不是一张损失曲面，而是一条带噪声、会漂移、受预算截断的梯度时间序列。怎样管理这条信号？" duration="建议 240–320 分钟" :outcomes="['从梯度序列逐项推导 Adam 的一阶矩、二阶矩与偏差修正', '解释 Adam + L2 与 AdamW 的状态差异，而不把二者写成同义词', '按固定 step 或 token 预算比较 warmup、cosine 与 one-cycle', '计算优化器状态显存，并为真实训练编写可审计参数组']" />

机器学习课已经讲过 SGD、Momentum、梯度下降与随机优化。本章不重复“梯度指向最陡上升方向”，而是研究现代深层网络真正交给优化器的接口：第 $t$ 步的随机梯度 $g_t$、参数状态 $\theta_t$、数值格式、学习率和剩余训练预算。

Adam 不是“永远比 SGD 快”的开关，AdamW 也不是把配置名改一下。它们分别回答两个更窄的问题：怎样从带噪梯度序列估计方向与尺度；怎样在不污染自适应矩状态的前提下对参数做衰减。

## 贯穿案例：固定 2 亿 token 的小型语言模型预训练

团队要在单张 24 GiB GPU 上训练一个 1.3 亿参数的 decoder-only 模型。训练预算不是“直到收敛”，而是固定为 2 亿个 token。每个优化步读取 8192 token，因此最多只有

$$
T=\frac{2\times10^8}{8192}\approx 24414
$$

次参数更新。比较优化器时必须固定 token 数、数据顺序、有效 batch、精度格式、梯度累积、模型初始化和评价时点。否则“收敛更快”可能只是多看了数据或多做了更新。

训练合同至少记录：

| 项目 | 本章基线 | 为什么必须记录 |
|---|---|---|
| 模型参数 | 130M，pre-norm Transformer | 决定梯度统计与状态内存 |
| 有效 batch | 8192 token/step | 决定梯度噪声尺度 |
| 预算 | 200M token | 调度曲线必须绑定结束点 |
| 优化器 | AdamW，$\beta_1=0.9,\beta_2=0.95$ | $\beta_2$ 会改变二阶矩响应速度 |
| 精度 | BF16 forward/backward，FP32 optimizer state | 区分计算格式与状态格式 |
| 梯度处理 | global norm clip = 1.0 | 防止偶发尖峰吞掉状态 |
| 评价 | 每 200 step 固定验证集 | 避免挑选偶然最低点 |

<DlQuiz kind="实验合同" question="两次实验都训练 10000 step，但实验 B 的 batch 是实验 A 的两倍。能否把验证损失差异只归因于优化器？" :options="['可以，step 数相同即可', '不可以，B 看过的 token 更多且梯度噪声尺度不同', '可以，只要随机种子相同', '只在 AdamW 下可以']" :answer="1" explanation="固定 step 不等于固定数据预算。有效 batch 同时改变已消费 token 数和随机梯度方差，因此至少要固定 token 预算与 batch 合同。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>若梯度累积从 1 改为 8、micro-batch 不变，哪些量改变：参数更新次数、每次更新看到的 token、优化器状态大小、调度的 step 计数？先写出四项再继续。</div>

## 1. Adam 把梯度当作时间序列

令当前 mini-batch 梯度为

$$
g_t=\nabla_\theta \ell_t(\theta_{t-1}).
$$

Adam 维护两个逐参数状态。一阶指数滑动平均追踪方向：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t.
$$

二阶原始矩追踪平方尺度：

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

这里的平方和乘法都是逐坐标操作。$v_t$ 不是梯度方差；若要得到中心化方差还需减去均值平方。Adam 使用未中心化二阶矩，是为了构造稳定的逐坐标尺度。

### 1.1 时间常数比“记忆长短”更准确

展开一阶矩可得

$$
m_t=(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i,
$$

旧梯度的权重按指数衰减。常用的粗略有效窗口是 $1/(1-\beta)$：$\beta_1=0.9$ 对应约 10 步，$\beta_2=0.999$ 对应约 1000 步。但它不是硬窗口；1000 步之前的梯度并不会突然消失。

当训练早期的梯度尺度迅速变化时，过大的 $\beta_2$ 会让 $v_t$ 对新尺度反应迟缓；当 batch 很小、梯度噪声很大时，过小的 $\beta_2$ 又会让分母剧烈抖动。选择 $\beta_2$ 要结合模型、batch、序列长度和梯度尖峰，而不是机械复制默认值。

<DlQuiz kind="状态预测" question="把 β₂ 从 0.999 降到 0.95，二阶矩状态通常会怎样？" :options="['对最近梯度平方响应更快，但估计更抖', '记忆更长且更平滑', '完全不影响更新', '只改变权重衰减']" :answer="0" explanation="较小 β₂ 降低旧状态权重，使 v 更快跟随尺度变化，同时减少时间平均，因此随机波动通常更大。" />

### 1.2 为什么需要偏差修正

若 $m_0=v_0=0$，训练初期的指数平均被零初始化拉小。假设梯度均值在短时间内近似为 $\mu$，则

$$
\mathbb E[m_t]=(1-\beta_1^t)\mu.
$$

所以用

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}
$$

修正初始化偏差。Adam 更新为

$$
\theta_t=\theta_{t-1}-\eta_t
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

偏差修正不是“让梯度变大”的经验技巧，而是补偿确定的零初始化因子。第 1 步若梯度为非零常数，修正后 $\widehat m_1=g_1$、$\widehat v_1=g_1^2$。

下面用一个标量梯度序列验证递推：

```python
import numpy as np

gradients = np.array([0.8, 0.5, -0.1, 1.0], dtype=np.float64)
beta1, beta2 = 0.9, 0.99
m = v = 0.0

for t, gradient in enumerate(gradients, start=1):
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    assert np.isfinite([m_hat, v_hat]).all()
    print(t, round(m_hat, 6), round(v_hat, 6))
```

### 1.3 $\epsilon$ 在哪里会改变算法

常见实现把 $\epsilon$ 放在平方根之后：

$$
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

也有实现计算 $\widehat m_t/\sqrt{\widehat v_t+\epsilon}$。当 $\widehat v_t$ 远大于 $\epsilon$ 时差异很小；在稀疏、小梯度或低精度区域，两者可能明显不同。复现实验必须登记优化器实现和 $\epsilon$ 位置，不能只写“Adam”。

<AdamWStateLab />

## 2. L2 正则化为什么不等于 AdamW

普通 L2 正则把

$$
\frac{\lambda}{2}\lVert\theta\rVert_2^2
$$

加到目标函数。它的梯度是 $\lambda\theta$，因此交给优化器的是

$$
g_t^{\mathrm{L2}}=g_t+\lambda\theta_{t-1}.
$$

在普通 SGD 中：

$$
\theta_t=\theta_{t-1}-\eta(g_t+\lambda\theta_{t-1})
=(1-\eta\lambda)\theta_{t-1}-\eta g_t,
$$

它等价于一次乘法权重衰减。但在 Adam 中，$g_t+\lambda\theta$ 同时进入 $m_t$ 与 $v_t$，之后还会被逐坐标除以 $\sqrt{\widehat v_t}$。衰减强度因此受到历史梯度尺度影响，不再是统一的 $(1-\eta\lambda)$。

AdamW 把两条路径分开：

$$
\theta_t=\theta_{t-1}
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}
-\eta_t\lambda\theta_{t-1}.
$$

任务梯度决定矩状态，权重衰减直接作用于参数。这样 $\lambda$ 的含义更清晰，也更容易与学习率共同调节。

<DlQuiz kind="概念辨析" question="Adam + L2 与 AdamW 的关键差异在哪里？" :options="['AdamW 不使用梯度', 'L2 项会进入 Adam 的一阶和二阶矩，AdamW 将参数衰减与自适应梯度路径分开', 'AdamW 没有学习率', '二者在所有优化器上完全等价']" :answer="1" explanation="自适应分母破坏了普通 SGD 下的简单等价关系。AdamW 的解耦让矩状态不包含 λθ。" />

### 2.1 哪些参数通常不衰减

现代模型常把矩阵权重放入 decay 组，把偏置和归一化尺度放入 no-decay 组。这不是数学定律，而是需要写清的建模选择。最危险的实现是按名称模糊匹配后静默漏掉参数。

```python
def split_weight_decay(model):
    decay, no_decay = [], []
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if parameter.ndim < 2 else decay
        target.append(parameter)
        seen.add(id(parameter))

    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert seen == trainable
    assert not ({id(p) for p in decay} & {id(p) for p in no_decay})
    return decay, no_decay
```

边界处应断言：每个可训练参数恰好进入一个组；冻结参数没有进入优化器；同一参数没有重复；checkpoint 恢复后参数组顺序不变。

<div class="dl-stop"><strong>停下来推一推 2</strong>若学习率按 cosine 降到接近 0，而 AdamW 的衰减项写成 $-\eta_t\lambda\theta$，训练末期的实际衰减强度怎样变化？这说明比较 $\lambda$ 时还必须固定什么？</div>

## 3. 学习率调度是一份预算分配合同

学习率决定每个优化步能移动多远。调度则把有限移动预算分配到训练早期、中期和末期。它不能脱离总 step 或 token 数讨论。

### 3.1 Warmup

线性 warmup 可写成

$$
\eta_t=\eta_{\max}\frac{t}{T_w},\qquad 0\le t\le T_w.
$$

训练刚开始时，参数随机、激活统计未稳定，Adam 的矩状态也只有很少观测。直接使用峰值学习率容易产生一次不可逆的巨大更新。Warmup 给状态估计和表示尺度留出建立时间，但过长会浪费固定预算。

### 3.2 Cosine decay

Warmup 之后常用

$$
\eta_t=\eta_{\min}+
\frac12(\eta_{\max}-\eta_{\min})
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right].
$$

$T$ 是计划结束步。若训练中途把总预算从 $T$ 改成 $2T$，同一个 step 的学习率也会改变。恢复训练时必须恢复 scheduler 状态，而不是只恢复模型参数。

### 3.3 One-cycle 与常数学习率

One-cycle 先升到峰值再下降，强调在有限预算中先探索再收敛；许多大模型训练也使用 warmup 后近似常数或 cosine。不存在脱离任务的“最佳曲线”。有意义的比较要固定：

1. 总 token 与更新次数；
2. 峰值和最小学习率；
3. warmup 占比；
4. batch 与梯度累积；
5. 评价 checkpoint；
6. 多随机种子或置信区间。

<ScheduleBudgetLab />

<DlQuiz kind="曲线判断" question="把总训练预算从 10000 步延长到 20000 步，但仍在第 10000 步比较 cosine 调度结果，会发生什么？" :options="['两条曲线必然完全相同', '延长预算的曲线在第 10000 步尚未衰减到终点，比较时学习率历史不同', '只影响权重衰减，不影响学习率', 'cosine 与预算无关']" :answer="1" explanation="cosine 的相位由 t/T 决定。改变 T 就改变了整个学习率历史，因此第 10000 步不是等价评价点。" />

## 4. 优化器状态也要进显存账本

设模型有 $N$ 个参数。若权重、梯度、一阶矩和二阶矩都以 FP32 保存，仅这四项就需要

$$
4N\times4\text{ bytes}=16N\text{ bytes}.
$$

混合精度训练还可能保存低精度模型权重和 FP32 master weight，使每参数状态达到 16–20 bytes，具体取决于框架和精度策略。130M 参数仅优化器相关状态就可能超过 1.5 GiB；更大的模型必须使用分片、offload 或不同状态格式。第 31 章会用这个账本推导 ZeRO 和 FSDP。

下列函数只计算声明过的状态，不假装给出端到端峰值：

```python
def optimizer_state_gib(parameters, bytes_per_parameter):
    if parameters <= 0 or bytes_per_parameter <= 0:
        raise ValueError("parameters and bytes_per_parameter must be positive")
    return parameters * bytes_per_parameter / 1024**3

print(optimizer_state_gib(130_000_000, 16))  # 约 1.94 GiB
```

端到端显存还包括激活、临时 workspace、通信 buffer、allocator 碎片和框架开销。报告“理论能放下”之前必须测量峰值。

## 5. 正确的训练与恢复顺序

一个可审计训练步通常按以下顺序执行：

1. 读取并校验 batch，记录有效 token 数；
2. forward 得到 loss；
3. 按梯度累积规则缩放 loss；
4. backward 累积梯度；
5. 必要时 unscale；
6. 检查非有限梯度并做 global norm clipping；
7. optimizer step；
8. scheduler step；
9. 清梯度；
10. 记录 loss、学习率、梯度范数、吞吐和显存。

框架对 scheduler 的调用时点可能不同，必须以所用实现为准。Checkpoint 至少要同时保存模型、优化器、scheduler、随机数状态、数据游标、全局 step、已消费 token、精度 scaler 和配置指纹。

<DlQuiz kind="恢复诊断" question="只恢复模型权重，不恢复 AdamW 的 m、v 和 scheduler，会怎样？" :options="['完全等价于无缝续训', '优化器时间状态和学习率相位重置，后续轨迹不再等价', '只会改变日志文件名', '只影响推理，不影响训练']" :answer="1" explanation="AdamW 的矩与偏差修正依赖历史 step，scheduler 也依赖相位；丢失它们相当于在旧参数上启动一套新优化过程。" />

## 6. 失效模式与发布门禁

| 现象 | 不能直接下的结论 | 第一批证据 |
|---|---|---|
| Loss 突然变 NaN | “模型太大” | loss scale、梯度范数、首个非有限张量、输入范围 |
| 验证损失前期更好 | “新优化器更优” | 是否固定 token、预算、评价点和随机种子 |
| AdamW 权重范数更小 | “泛化一定更好” | 参数组、有效衰减积分、验证切片 |
| 恢复后 loss 跳变 | “数据随机性” | optimizer/scheduler/scaler/data cursor 是否恢复 |
| 更大 batch 吞吐更高 | “训练更快收敛” | 达到同一验证损失所需 token、时间和能耗 |

本章发布门禁：运行日志能重建每一步的有效 batch、token、学习率、梯度范数和参数组；任意 checkpoint 可在固定容差内恢复下一步 loss；优化器比较固定预算并报告至少三个种子。

<div class="dl-stop"><strong>停下来推一推 3</strong>一条运行记录显示“AdamW 比 Adam 最终验证损失低 0.03”。列出至少五项仍需核对的合同，才能把差异归因于解耦衰减。</div>

## 7. 推荐论文阅读

<DlPaperCard title="Adam: A Method for Stochastic Optimization" meta="Kingma & Ba · 2014" priority="必读" question="偏差修正从哪里来？算法 1 中的状态、时间步和 ε 分别在哪里进入更新？" reading="先读摘要与第 2 节，再逐行对照 Algorithm 1；最后看实验，不把论文中的默认超参数理解成跨任务定律。" href="https://arxiv.org/pdf/1412.6980" />

<DlPaperCard title="Decoupled Weight Decay Regularization" meta="Loshchilov & Hutter · 2017" priority="必读" question="为什么 L2 正则与权重衰减在自适应优化器中不再等价？" reading="重点读第 2 节的等价关系和第 2.1 节的解耦更新；阅读实验时核对学习率与衰减系数是否分别搜索。" href="https://arxiv.org/pdf/1711.05101" />

<DlPaperCard title="Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" meta="Ioffe & Szegedy · 2015" priority="选读 · 下一章入口" question="优化难度变化究竟来自输入分布稳定、尺度重参数化，还是其他机制？" reading="本章只读问题与算法接口；第 02 章再审计 batch 统计、训练/推理差异和后续解释边界。" href="https://arxiv.org/pdf/1502.03167" />

## 8. 练习

1. 手算梯度序列 $[1,1,1]$ 在 $\beta_1=0.9,\beta_2=0.99$ 下前三步的 $m_t,v_t,\widehat m_t,\widehat v_t$。
2. 构造两个坐标具有不同梯度尺度的例子，比较 SGD 与 Adam 的单步更新方向和长度。
3. 对同一标量参数实现 Adam + L2 与 AdamW，验证当 $\lambda=0$ 时轨迹一致、$\lambda>0$ 时矩状态不同。
4. 为 130M 模型写出 FP32、FP16+master weight、BF16 三种假设下的状态显存账本，明确哪些项是框架相关的。
5. 固定总 token，比较 warmup+constant 与 warmup+cosine；报告最终值之外的学习率积分和达到目标损失的 token 数。
6. 设计 checkpoint 恢复测试：训练 20 步保存，继续 1 步；从 checkpoint 恢复再运行同一 batch，比较 loss、梯度和参数。

## 本章小结

- Adam 用一阶矩追踪方向、二阶原始矩追踪尺度；偏差修正补偿零初始化。
- $\beta_1,\beta_2,\epsilon$ 都有时间和数值含义，不能只写优化器名称。
- Adam + L2 会把 $\lambda\theta$ 写入自适应矩；AdamW 把权重衰减从梯度路径解耦。
- 学习率调度绑定训练预算，比较时必须固定 token、batch、峰值和评价时点。
- 优化器状态、scheduler、scaler、数据游标和 RNG 都是可恢复训练的一部分。

<DlLessonProgress lesson-id="01" next-href="/courses/deep-learning/02-stable-training" next-label="下一章：稳定训练" />
