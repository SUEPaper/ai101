---
course: deep-learning
edition: industrial
title: VAE、VQ-VAE 与自回归生成
description: 从变分下界与重参数化进入离散码本、commitment loss 和自回归因子分解，比较似然、重构、采样速度与表示容量。
outline: [2, 3]
---

<DlLessonHero number="14" module="生成模型" title="VAE、VQ-VAE 与自回归生成" question="若生成器要同时编码样本、计算训练目标并从先验采样，连续潜变量、离散 token 与自回归顺序各自承担什么？" duration="建议 300–390 分钟" :outcomes="['从边缘似然推导 ELBO 与 KL gap', '实现 Gaussian 重参数化并诊断 posterior collapse', '解释 VQ-VAE 最近邻量化、直通估计和码本坍缩', '比较 PixelCNN/Transformer 自回归似然与串行采样成本']" />

## 贯穿案例：压缩并生成高分辨率缺陷纹理

目标是把 256×256 图像压成较短潜变量，再生成新纹理。VAE 提供连续潜空间和近似似然；VQ-VAE 把图像变成离散 code；自回归 prior 对 code 序列建模。评价同时包含 rate、distortion、感知质量、码本利用率和采样延迟。

<DlQuiz kind="模型边界" question="普通 autoencoder 的重构损失是否自动定义了可从中采样的潜变量先验？" :options="['自动定义', '不定义，未约束的 latent 分布通常没有明确采样规则', '只要 decoder 足够大就定义', '只对图像定义']" :answer="1" explanation="VAE 的 KL 项或其他先验匹配机制才把编码分布连接到可采样先验。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>若训练重构很好，但从 N(0,I) 采样全是垃圾，画出编码 latent 与先验可能怎样错位。</div>

## 1. ELBO 从不可解边缘似然来

潜变量模型 $p_\theta(x,z)=p(z)p_\theta(x|z)$ 的边缘似然包含积分。引入近似后验 $q_\phi(z|x)$：

$$
\log p_\theta(x)=\underbrace{\mathbb E_q\log p_\theta(x|z)-KL(q_\phi(z|x)||p(z))}_{ELBO}
+KL(q_\phi(z|x)||p_\theta(z|x)).
$$

最后一项非负，所以最大化 ELBO 是最大化下界。重构项与输出分布绑定：Bernoulli、Gaussian、discretized logistic 会产生不同像素损失，不能把 MSE 当作无假设选择。

```python
import numpy as np

mean = np.array([0.5, -0.2])
logvar = np.array([-0.7, 0.3])
kl = 0.5 * np.sum(mean**2 + np.exp(logvar) - 1.0 - logvar)
assert kl >= 0
assert np.isclose(kl, 0.5 * np.sum(mean**2 + np.exp(logvar) - 1 - logvar))
```

<VariationalBottleneckLab />

<DlQuiz kind="ELBO" question="ELBO 与真实 log p(x) 的差是什么？" :options="['重构误差', 'q(z|x) 与真实后验 p(z|x) 的 KL', '优化器动量', '码本大小']" :answer="1" explanation="变分后验越接近真实后验，下界越紧。" />

## 2. 重参数化把随机节点变成可微路径

Gaussian encoder 输出 $\mu,\log\sigma^2$，采样改写为

$$z=\mu+\sigma\odot\epsilon,\quad \epsilon\sim\mathcal N(0,I).$$

随机性移到与参数无关的 $\epsilon$，梯度可通过 $\mu,\sigma$。使用 log variance 避免直接预测负方差；低精度下还要限制极端 logvar。

```python
import numpy as np

rng = np.random.default_rng(7)
mu = np.array([0.3, -0.4])
logvar = np.array([-0.2, 0.6])
epsilon = rng.normal(size=mu.shape)
z = mu + np.exp(0.5 * logvar) * epsilon
assert z.shape == mu.shape and np.isfinite(z).all()
```

<DlQuiz kind="重参数化" question="为什么预测 log variance 而非直接预测 variance？" :options="['让方差可以为负', '指数变换保证正值并改善尺度参数化', '删除随机性', '只为节省一个参数']" :answer="1" explanation="log-space 能表示大范围正方差；仍需数值裁剪。" />

## 3. Posterior collapse 与 β-VAE

强 decoder 可能忽略 $z$，使 $q(z|x)\approx p(z)$、KL 接近 0。应看每维 KL、active units、mutual-information 代理和 decoder 对打乱 $z$ 的敏感性。KL warmup、free bits、削弱 decoder 或改变目标可缓解。

$\beta$-VAE 用 $\beta KL$ 加强先验约束，可能获得更规则表示，也会降低重构容量。“disentanglement”必须声明监督与度量，不能只靠潜变量插值挑图。

<DlQuiz kind="坍缩诊断" question="KL 接近 0 且打乱 z 不改变重构，最可能是什么？" :options="['码本利用率完美', 'decoder 忽略 latent 的 posterior collapse', 'FID 必然为零', '训练数据重复']" :answer="1" explanation="后验退回先验且输出不依赖 z，说明潜变量通道未被使用。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>KL=0 不一定总是 bug。若数据本身无信息或某些 latent 维不需要，怎样按维度和样本区分合理闲置与整体坍缩？</div>

## 4. VQ-VAE：把 latent 变成离散 code

encoder 输出 $z_e(x)$，从码本 $\{e_k\}$ 选最近向量 $k^*=\arg\min_k\lVert z_e-e_k\rVert^2$。不可微 argmin 用 straight-through estimator 把 decoder 梯度近似传回 encoder；codebook loss 更新向量，commitment loss 防止 encoder 输出漂移。

<CodebookAutoregressiveLab />

码本 perplexity 反映使用分布的有效 code 数。少数 code 垄断是 codebook collapse；可用 EMA 更新、重启死码、调整 commitment 或数据/encoder 容量，但必须保持重构和 prior 难度的联合账本。

```python
import numpy as np

encoder_value = np.array([0.8, -0.1])
codebook = np.array([[0., 0.], [1., 0.], [0., 1.]])
distances = ((codebook - encoder_value) ** 2).sum(axis=1)
index = int(np.argmin(distances))
quantized = codebook[index]
assert index == 1 and np.array_equal(quantized, [1., 0.])
```

## 5. 自回归先验：精确因子分解，串行生成

对离散序列 $x_{1:T}$：

$$p(x)=\prod_{t=1}^T p(x_t|x_{<t}).$$

PixelCNN 用 masked convolution 保证像素因果顺序；Transformer 用 causal attention 建模 code token。训练可并行计算所有位置的 teacher-forced loss，采样必须逐 token，除非使用缓存、并行近似或其他生成策略。

<DlQuiz kind="采样成本" question="自回归模型为何训练可并行、标准采样却串行？" :options="['训练知道完整前缀，可同时计算各位置条件损失；采样时下一 token 依赖已生成前缀', '训练没有因果 mask', '采样不使用模型', '只因 GPU 太慢']" :answer="0" explanation="因果分解决定了标准 ancestral sampling 的数据依赖。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>把 256×256 像素自回归改为 32×32 VQ code 自回归，序列长度、码本词表、重构误差和 prior 难度怎样交换？</div>

## 6. 推荐论文阅读

<DlPaperCard title="Auto-Encoding Variational Bayes" meta="Kingma & Welling · 2013" priority="必读 · VAE" question="ELBO 与重参数化分别解决统计与梯度的什么问题？" reading="精读变分下界、SGVB estimator 与 Gaussian 实例。" href="https://arxiv.org/pdf/1312.6114" />

<DlPaperCard title="Neural Discrete Representation Learning" meta="van den Oord et al. · 2017" priority="必读 · VQ-VAE" question="最近邻量化、直通梯度和 learned prior 如何组成离散生成系统？" reading="读三项 loss、码本更新、PixelCNN prior 与消融。" href="https://arxiv.org/pdf/1711.00937" />

<DlPaperCard title="Pixel Recurrent Neural Networks" meta="van den Oord et al. · 2016" priority="选读 · 自回归" question="二维像素顺序如何转为 masked spatial dependency，精确 likelihood 付出什么采样成本？" reading="重点读条件分解、Row/Diagonal LSTM 与采样过程。" href="https://arxiv.org/pdf/1601.06759" />

## 7. 练习与小结

1. 推导对角 Gaussian KL；2. 监控 per-dim KL 与 active units；3. 注入死码并设计重启；4. 比较像素与 VQ token 序列长度；5. 报告 rate–distortion–perception–latency 四轴。

## 本章小结

- VAE 最大化可计算 ELBO，重构项隐含输出概率模型。
- 重参数化建立低方差可微路径；KL 过强与强 decoder 可导致 posterior collapse。
- VQ-VAE 用离散码本连接压缩表示与自回归 prior，码本利用率是核心状态。
- 自回归因子分解提供明确 likelihood，但标准采样受序列依赖限制。

<DlLessonProgress lesson-id="14" next-href="/courses/deep-learning/15-diffusion-flow-matching" next-label="下一章：Diffusion 与 Flow Matching" />
