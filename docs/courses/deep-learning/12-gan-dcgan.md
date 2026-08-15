---
course: deep-learning
edition: industrial
title: GAN 与 DCGAN
description: 从密度比判别、minimax 与非饱和生成器目标出发，建立 DCGAN 形状、交替优化、模式坍缩与训练证据合同。
outline: [2, 3]
---

<DlLessonHero number="12" module="生成模型" title="GAN 与 DCGAN" question="当模型没有显式似然、训练目标又随对手变化时，怎样判断生成器真的覆盖了数据分布？" duration="建议 280–360 分钟" :outcomes="['从最优判别器推导 GAN 与密度比、JS divergence 的关系', '区分原始 minimax 与非饱和生成器损失的梯度', '写出 DCGAN 逐层 shape 与交替优化合同', '用覆盖率、最近邻和固定样本诊断模式坍缩与记忆']" />

生成模型学习数据分布并产生新样本。GAN 不显式写出归一化密度，而让生成器 $G(z)$ 与判别器 $D(x)$ 对抗：$D$ 区分真实与生成，$G$ 让生成样本被判为真实。它的强项是锐利样本，难点是训练不是固定目标的普通最小化。

## 贯穿案例：稀有表面缺陷合成

团队想合成 64×64 的裂纹、孔洞和污染纹理来补充检测训练。真实数据只有 6000 张且类别不平衡。发布标准不允许只展示“最好看的 20 张”：必须检查每类覆盖、近重复、训练集最近邻、条件一致性和增强后下游收益。

<DlQuiz kind="目标边界" question="GAN 生成样本锐利是否证明它覆盖了全部训练分布？" :options="['证明', '不证明，模型可能只覆盖少数模式或记忆训练样本', '只要 D loss 为零就证明', '只在 DCGAN 中证明']" :answer="1" explanation="视觉质量与分布覆盖是不同维度，必须用多样性和最近邻证据补充。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>若三类缺陷中生成器只产生最常见的裂纹，判别器和总体视觉评分可能怎样？设计一个按类别与形态覆盖的检查。</div>

## 1. 对抗目标与最优判别器

经典 minimax 目标为

$$
\min_G\max_D V(D,G)=\mathbb E_{x\sim p_{data}}\log D(x)+
\mathbb E_{z\sim p(z)}\log(1-D(G(z))).
$$

固定 $G$，对每个 $x$ 最大化 $a\log D+b\log(1-D)$，一阶条件给出

$$
D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}.
$$

因此 $D/(1-D)$ 估计密度比。代回目标可得到与 Jensen–Shannon divergence 的关系，理想平衡是 $p_g=p_{data}$、$D=1/2$。但高维早期两分布支持几乎不重叠时，JS 可能饱和，理论固定点不代表优化路径良好。

```python
import numpy as np

data = np.array([0.6, 0.3, 0.1], dtype=np.float64)
generated = np.array([0.2, 0.5, 0.3], dtype=np.float64)
optimal_d = data / (data + generated)
density_ratio = optimal_d / (1.0 - optimal_d)
assert np.allclose(density_ratio, data / generated)
assert np.all((0 < optimal_d) & (optimal_d < 1))
```

<GanGameLab />

<DlQuiz kind="固定点" question="当 p_g=p_data 且判别器最优时，D(x) 应是多少？" :options="['0', '0.5', '1', '随类别数变化']" :answer="1" explanation="真实和生成来源后验各占一半，最优判别器无法区分。" />

## 2. 生成器为何常用非饱和目标

原始 minimax 让 $G$ 最小化 $\log(1-D(G(z)))$。训练早期若 $D(G(z))\approx0$，sigmoid 饱和会给生成器弱信号。实践常改为

$$
L_G^{NS}=-\mathbb E_z\log D(G(z)),
$$

它不改变理想固定点，却在判别器自信拒绝假样本时提供更强梯度。这里“更强”不是越大越好：判别器过拟合、小 batch 统计泄漏和不平衡更新仍会导致震荡。

```python
import numpy as np

probabilities = np.array([1e-4, 0.1, 0.5], dtype=np.float64)
minimax_signal = probabilities * (1 - probabilities)
nonsaturating_signal = 1 - probabilities
assert nonsaturating_signal[0] > minimax_signal[0] * 1000
```

<DlQuiz kind="损失辨析" question="非饱和生成器损失的主要实践理由是什么？" :options="['它让判别器消失', '判别器很强时仍给生成器较大的学习信号', '它等于最大似然', '它保证不坍缩']" :answer="1" explanation="非饱和目标改善早期梯度，但不保证覆盖或收敛。" />

## 3. 交替优化是一份状态合同

每个迭代通常先更新 $D$，再更新 $G$。更新 $D$ 时应阻断假样本到 $G$ 的梯度；更新 $G$ 时冻结 $D$ 参数但仍需通过 $D$ 对输入求梯度。训练记录至少包含：D/G 更新比、优化器状态、真实与假 batch、增强、固定 latent、随机 latent、EMA 生成器与 checkpoint 父版本。

```python
def update_schedule(iterations, critic_steps=1):
    events = []
    for step in range(iterations):
        events.extend((step, "D") for _ in range(critic_steps))
        events.append((step, "G"))
    return events

events = update_schedule(3, critic_steps=2)
assert [kind for _, kind in events].count("D") == 6
assert [kind for _, kind in events].count("G") == 3
```

<div class="dl-stop"><strong>停下来推一推 2</strong>更新生成器时若忘记冻结 D 参数，数值输出可能仍“正常”。哪些梯度、优化器状态和显存观测能发现这个错误？</div>

## 4. DCGAN 的结构约束

DCGAN 用 stride convolution 取代 pooling，生成器用转置卷积逐级放大；通常移除隐藏全连接层，生成器用 BN+ReLU、输出 tanh，判别器用 LeakyReLU。输入图像若映射到 $[-1,1]$，生成器输出 tanh 才与数据合同一致。

<DcganShapeLab />

每层必须记录 NCHW、kernel、stride、padding 和输出。棋盘格伪影来自转置卷积不均匀重叠时，可改用 resize-conv；BatchNorm 在 batch 很小或 D/G 统计混杂时也可能不稳。

<DlQuiz kind="数据接口" question="生成器输出使用 tanh 时，真实图像最常见的匹配归一化是什么？" :options="['保持 0–255 整数', '缩放到约 [-1,1]', '只减去类别均值', '转成 one-hot']" :answer="1" explanation="真实与生成输入 D 的数值范围应一致。" />

## 5. 模式坍缩与训练指纹

模式坍缩表现为许多 $z$ 映射到少数样本。D/G loss 不能单独诊断：两者是随对手变化的相对目标。需要同时保存固定 latent 网格观察轨迹、随机样本看覆盖、特征空间聚类、类别条件直方图、训练集最近邻与下游性能。

常见干预包括调整更新比、正则化 D、谱归一化、minibatch discrimination、历史平均或改用 Wasserstein 目标。每个干预都要用同一固定 latent 与覆盖指标验证，不以“loss 更平滑”代替。

<DlQuiz kind="诊断" question="固定 latent 网格与每次重新采样的网格分别回答什么？" :options="['完全相同', '固定网格观察同一样本的训练轨迹；随机网格观察当前分布覆盖', '固定网格只测速度', '随机网格只测参数量']" :answer="1" explanation="二者要一起使用，避免轨迹稳定被误读为分布覆盖。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>生成图像与训练集最近邻几乎一致，但 FID 很低。说明 FID 为什么不能单独排除记忆，并设计隐私/重复检查。</div>

## 6. 推荐论文阅读

<DlPaperCard title="Generative Adversarial Nets" meta="Goodfellow et al. · 2014" priority="必读 · 起点" question="最优判别器、JS divergence 与 minimax 固定点如何连接，理论为何未保证训练路径？" reading="精读命题 1、定理 1 和算法框；逐项区分密度结论与参数优化。" href="https://arxiv.org/pdf/1406.2661" />

<DlPaperCard title="Unsupervised Representation Learning with Deep Convolutional GANs" meta="Radford, Metz & Chintala · 2015" priority="必读 · 结构" question="哪些卷积结构规则稳定了训练，表示插值和特征算术提供了什么证据？" reading="读架构指南、训练细节、最近邻与表示可视化，不只看样本图。" href="https://arxiv.org/pdf/1511.06434" />

<DlPaperCard title="Improved Techniques for Training GANs" meta="Salimans et al. · 2016" priority="选读 · 诊断工具" question="feature matching、minibatch discrimination 与 inception score 分别针对什么失效？" reading="按技术逐项读，不把多个技巧组合结果归因于单一部件。" href="https://arxiv.org/pdf/1606.03498" />

## 7. 练习与门禁

1. 手推离散三点分布的最优 $D$ 和 GAN value。
2. 对极小 $D(G(z))$ 比较两种生成器目标的 logit 梯度。
3. 为 4→8→16→32→64 DCGAN 写逐层 shape 断言。
4. 注入 mode collapse，比较固定 latent、随机覆盖和最近邻指纹。
5. 冻结数据切分，验证合成增强是否改善真实测试集而非只改善混合验证集。

## 本章小结

- GAN 用判别器估计真实/生成密度比，但理想固定点不保证可训练路径。
- 非饱和目标在强判别器下改善生成器信号，不保证消除模式坍缩。
- DCGAN 的归一化、激活与逐层 shape 是完整训练合同的一部分。
- D/G loss 是相对博弈信号，必须与固定样本、覆盖、最近邻和下游证据联合解释。

<DlLessonProgress lesson-id="12" next-href="/courses/deep-learning/13-wgan-stylegan" next-label="下一章：WGAN、StyleGAN 与生成评估" />
