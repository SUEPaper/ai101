---
course: deep-learning
edition: industrial
title: WGAN、StyleGAN 与生成质量评估
description: 从 Wasserstein 距离、1-Lipschitz critic 与梯度惩罚进入 StyleGAN 的映射网络、调制卷积和层级控制，再建立 FID/KID、覆盖与记忆评价矩阵。
outline: [2, 3]
---

<DlLessonHero number="13" module="生成模型" title="WGAN、StyleGAN 与生成质量评估" question="更平滑的 critic 目标怎样改善分布不重叠时的信号；更好看的样本又该用什么证据证明不是挑图、坍缩或记忆？" duration="建议 290–380 分钟" :outcomes="['解释 Wasserstein-1 距离与 1-Lipschitz 对偶', '比较 weight clipping、gradient penalty 与谱归一化', '拆解 StyleGAN 的 z→w、调制卷积、噪声与 truncation', '正确使用 FID、KID、precision/recall 和最近邻审计']" />

## 贯穿案例：多规格缺陷生成器资格审查

生成器要覆盖 12 种规格、4 类缺陷和多个相机域。团队展示的样本很锐利，但小规格几乎不出现。我们将把“质量”拆成保真、覆盖、条件一致、非记忆、公平切片和下游效用。

<DlQuiz kind="评价边界" question="只挑选人工认为最好看的样本能否证明生成分布质量？" :options="['可以', '不可以，选择过程隐藏失败率与覆盖缺口', '样本超过 10 张即可', '只对 StyleGAN 可以']" :answer="1" explanation="样本选择协议必须固定或随机，评价还要覆盖整个生成分布。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>写一个盲评协议：随机种子、样本数、条件分层、真实/生成混排、重复样本与评审一致性怎样处理？</div>

## 1. Wasserstein 距离提供连续几何

Earth Mover 距离可理解为把 $p_g$ 搬到 $p_{data}$ 的最小运输成本。Kantorovich–Rubinstein 对偶写成

$$
W_1(p_r,p_g)=\sup_{\lVert f\rVert_L\le1}\mathbb E_{p_r}f(x)-\mathbb E_{p_g}f(x).
$$

$f$ 是 critic，不是概率分类器。1-Lipschitz 约束保证函数斜率有界，使分布支持不重叠时仍可能提供有意义方向。

```python
import numpy as np

real = np.array([-1.0, 0.0, 2.0])
fake = np.array([-2.0, -1.0, 1.0])
# 一维等权经验分布的 W1 等于排序后逐点距离均值
w1 = np.abs(np.sort(real) - np.sort(fake)).mean()
assert np.isclose(w1, 1.0)
```

<WassersteinCriticLab />

<DlQuiz kind="Critic 接口" question="WGAN critic 输出端为何不使用 sigmoid？" :options="['因为输出类别很多', 'critic 学受 Lipschitz 约束的实值函数差，不是概率', '因为不反向传播', '因为只接收文本']" :answer="1" explanation="Wasserstein 对偶需要实值 critic；把它压成概率会改变函数类。" />

## 2. 怎样近似 Lipschitz 约束

原始 WGAN 将权重 clip 到小区间，简单却可能把 critic 压成低容量函数。WGAN-GP 在真实与生成样本之间的插值点加入

$$
\lambda\mathbb E_{\hat x}(\lVert\nabla_{\hat x}f(\hat x)\rVert_2-1)^2.
$$

谱归一化则约束每层权重谱范数。三者约束对象不同，都不是数学上精确覆盖所有空间的 Lipschitz 证书。

```python
import numpy as np

gradient_norms = np.array([0.7, 1.0, 1.4])
penalty = ((gradient_norms - 1.0) ** 2).mean()
assert np.isclose(penalty, (0.09 + 0.0 + 0.16) / 3)
```

<DlQuiz kind="正则化" question="WGAN-GP 通常在哪里采样梯度惩罚点？" :options="['只在参数空间', '真实与生成样本之间的插值点', '只在标签上', '只在优化器状态上']" :answer="1" explanation="插值点覆盖两分布之间 critic 最需要提供运输方向的区域。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>critic loss 持续下降但样本变差。列出更新比、GP 权重、数据增强、生成器容量和 EMA 五个可观测假设。</div>

## 3. StyleGAN：把采样空间与层级控制分开

StyleGAN 用映射网络把 $z$ 变为中间潜变量 $w$，再在各合成层调制特征统计或卷积权重。独立噪声提供随机微结构。StyleGAN2 用 modulated convolution 与 demodulation 修复 AdaIN 相关伪影。

层级早期通常影响姿态/布局，后期影响纹理/颜色，但不是严格可解释因果轴。style mixing 用两个 $w$ 在不同 cutoff 注入，检测网络是否把尺度层级组织起来。

<StyleMixLab />

truncation 把 $w$ 向平均 $\bar w$ 收缩：$w'=\bar w+\psi(w-\bar w)$。较小 $\psi$ 常提高典型样本质量、降低覆盖；评价时必须登记 $\psi$。

<DlQuiz kind="潜空间" question="StyleGAN 映射网络 z→w 的主要目的是什么？" :options="['把图像压成 JPEG', '让采样先验与合成层所需的样式控制空间解耦', '删除随机性', '直接输出类别']" :answer="1" explanation="w 不必保持简单先验几何，可更适合逐层调制。" />

## 4. FID、KID 与有限样本偏差

FID 在固定特征空间把真实与生成近似为高斯：

$$
FID=\lVert\mu_r-\mu_g\rVert^2+
\operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}).
$$

它同时响应质量与覆盖，但依赖特征网络、resize、样本数和数值实现，有限样本有偏。KID 用多项式核 MMD 的无偏估计，更适合报告置信区间。不同数据域使用 ImageNet 特征是否合理也要验证。

```python
import numpy as np

real = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
generated = real + np.array([0.5, -0.25])
mean_term = np.sum((real.mean(0) - generated.mean(0)) ** 2)
assert np.isclose(mean_term, 0.3125)
```

<DlQuiz kind="指标合同" question="比较两个 FID 时至少应固定什么？" :options="['只固定模型名', '特征网络、预处理、真实集、样本数与实现', '只固定随机种子', '只固定图片格式']" :answer="1" explanation="FID 对特征和有限样本协议敏感，跨协议数值不可直接比较。" />

## 5. 质量—覆盖—记忆三轴

生成 precision 衡量样本是否落在真实流形附近，recall 衡量真实分布是否被生成覆盖。二者可揭示“很真但单一”和“很多样但失真”。再加训练集最近邻、成员推断/重复率和条件一致性，才形成资格矩阵。

按规格、类别、相机、少数群体切片计算指标。总体 FID 可能被主域支配。下游效用应只在冻结真实测试集上验证，且比较真实数据量相同的基线。

<DlQuiz kind="覆盖判断" question="生成 precision 高、recall 低最可能表示什么？" :options="['样本普遍逼真但只覆盖少数模式', '样本覆盖广但都失真', '一定发生训练泄漏', '指标完全理想']" :answer="0" explanation="precision 与 fidelity 接近，recall 与分布覆盖接近。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>总体 FID 改善 15%，但少数规格 recall 减半。给出发布、拒绝或继续训练所需的业务权重与门禁。</div>

## 6. 推荐论文阅读

<DlPaperCard title="Wasserstein GAN" meta="Arjovsky, Chintala & Bottou · 2017" priority="必读 · 距离" question="弱拓扑与 Wasserstein 距离为何能改善分布支持不重叠时的连续信号？" reading="读理论动机、critic 训练和 loss 相关性；区分距离理论与 clipping 实现。" href="https://arxiv.org/pdf/1701.07875" />

<DlPaperCard title="Improved Training of Wasserstein GANs" meta="Gulrajani et al. · 2017" priority="必读 · 梯度惩罚" question="weight clipping 造成什么 critic 病态，插值点 GP 如何缓解？" reading="精读梯度范数图、算法和多架构对照。" href="https://arxiv.org/pdf/1704.00028" />

<DlPaperCard title="A Style-Based Generator Architecture for Generative Adversarial Networks" meta="Karras, Laine & Aila · 2018" priority="必读 · 层级控制" question="映射网络、style mixing、噪声与 truncation 分别控制什么统计？" reading="按 synthesis network 路径读，再核对 disentanglement 指标与主观样本的差异。" href="https://arxiv.org/pdf/1812.04948" />

## 7. 练习与门禁

1. 在一维经验分布上手算 W1，并构造合法 1-Lipschitz critic。
2. 比较 clipping、GP、谱归一化的约束位置与 profiler 成本。
3. 固定 style cutoff 生成成对样本，盲评粗/细属性是否真被分离。
4. 用相同 50k 样本、特征和 resize 比较 FID/KID，bootstrap KID 区间。
5. 同时报告 precision/recall、最近邻与每条件覆盖率。

## 本章小结

- WGAN 用 1-Lipschitz critic 估计 Wasserstein 距离，提供更连续的分布差异信号。
- clipping、GP 与谱归一化是不同近似约束，不能混写成同一算法。
- StyleGAN 用中间潜空间和逐层调制组织层级控制，truncation 交换质量与多样性。
- FID 依赖协议且有有限样本偏差，必须联合 KID、precision/recall、最近邻与切片证据。

<DlLessonProgress lesson-id="13" next-href="/courses/deep-learning/14-vae-vqvae-autoregressive" next-label="下一章：VAE、VQ-VAE 与自回归生成" />
