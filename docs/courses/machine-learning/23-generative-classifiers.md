---
title: 23. 生成式分类器怎样得到决策边界
description: 从类条件密度和 Bayes 公式推导 LDA、QDA 与朴素 Bayes
course: machine-learning
---

<MlLessonHero
  eyebrow="第六模块 · 概率与潜变量"
  title="生成式分类器怎样得到决策边界"
  subtitle="先描述每个类别如何生成特征，再用 Bayes 公式把生成故事翻译成分类规则。"
  :diagnostics="['能从高斯类条件分布推导 LDA/QDA 边界', '能解释共享协方差为何产生线性判别式', '能判断朴素条件独立假设的收益与代价']"
/>

## 1. 生成式与判别式的分界

生成式分类建模联合分布：

$$
p(\mathbf x,y)
=p(y)p(\mathbf x\mid y).
$$

预测时用 Bayes 公式：

$$
p(y=k\mid\mathbf x)
=\frac{\pi_kp(\mathbf x\mid y=k)}
{\sum_{\ell}\pi_\ell p(\mathbf x\mid y=\ell)},
$$

其中 $\pi_k=p(y=k)$ 是类别先验。判别式模型如逻辑回归直接建模 $p(y\mid\mathbf x)$，无需描述 $\mathbf x$ 的完整分布。

生成式模型能处理缺失结构、采样和先验变化，但错误的特征分布假设也可能浪费容量。

<MlLab
  kind="gda"
  title="实验：协方差怎样弯曲决策边界"
  description="调节两类均值、方差与相关性，观察 LDA 的直线和 QDA 的二次曲线。"
/>

## 2. 高斯判别分析

假设

$$
p(\mathbf x\mid y=k)
=\mathcal N(\mathbf x;\boldsymbol\mu_k,\boldsymbol\Sigma_k).
$$

忽略对所有类别相同的常数，类别判别分数为

$$
\delta_k(\mathbf x)
=\log\pi_k
-\frac12\log|\boldsymbol\Sigma_k|
-\frac12(\mathbf x-\boldsymbol\mu_k)^\top
\boldsymbol\Sigma_k^{-1}
(\mathbf x-\boldsymbol\mu_k).
$$

选择

$$
\hat y=\arg\max_k\delta_k(\mathbf x).
$$

这个分数同时考虑：

- 先验 $\pi_k$；
- 分布体积 $|\Sigma_k|$；
- 按协方差尺度计算的 Mahalanobis 距离。

<MlQuiz
  question="高斯判别分数中的 Mahalanobis 距离相比欧氏距离多考虑了什么？"
  :options="['特征尺度与相关性', '类别名称长度', '样本排列顺序', '优化器学习率']"
  :answer="0"
  explanation="协方差逆矩阵对不同方向按方差缩放，并处理特征相关。"
/>

## 3. LDA 为什么是线性边界

LDA 假设所有类别共享协方差：

$$
\boldsymbol\Sigma_k=\boldsymbol\Sigma.
$$

展开二次项：

$$
\begin{aligned}
&-\frac12(\mathbf x-\boldsymbol\mu_k)^\top
\boldsymbol\Sigma^{-1}
(\mathbf x-\boldsymbol\mu_k)\\
&=
-\frac12\mathbf x^\top\boldsymbol\Sigma^{-1}\mathbf x
+\mathbf x^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_k
-\frac12\boldsymbol\mu_k^\top
\boldsymbol\Sigma^{-1}\boldsymbol\mu_k.
\end{aligned}
$$

第一项对所有类别相同，比较类别时抵消；$\log|\Sigma|$ 也相同。因此

$$
\delta_k^{\rm LDA}(\mathbf x)
=\mathbf x^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_k
-\frac12\boldsymbol\mu_k^\top
\boldsymbol\Sigma^{-1}\boldsymbol\mu_k
+\log\pi_k.
$$

任意两类令 $\delta_k=\delta_\ell$，得到 $\mathbf x$ 的线性方程，所以边界是超平面。

::: ml-stop 停下来推一推 1
一维两类方差相同、先验相同、均值为 $\mu_0,\mu_1$。证明边界在 $(\mu_0+\mu_1)/2$。
:::

<MlQuiz
  question="LDA 的决策边界为何是线性的？"
  :options="['所有均值相同', '共享协方差使各类共同的 x 二次项比较时抵消', '高斯分布没有平方项', '先验必须相等']"
  :answer="1"
  explanation="展开高斯对数密度后，x^TΣ^-1x 对所有类别相同，剩下关于 x 的一次项。"
/>

## 4. QDA 为什么是二次边界

QDA 允许每类独立协方差 $\Sigma_k$。此时

$$
-\frac12\mathbf x^\top\Sigma_k^{-1}\mathbf x
$$

不能在类别间抵消，令两类判别分数相等后保留二次项，得到椭圆、双曲线等二次曲面。

自由度代价很大：每个 $d\times d$ 对称协方差有 $d(d+1)/2$ 个独立参数。小样本高维时协方差可能奇异。常见折中是收缩：

$$
\widetilde\Sigma_k
=(1-\lambda)\hat\Sigma_k+\lambda\alpha\mathbf I.
$$

$\lambda=0$ 接近 QDA，强共享或球形约束降低方差。

<MlQuiz
  question="相对 LDA，QDA 的主要交换是什么？"
  :options="['更灵活的二次边界，但需估计更多协方差参数', '只能产生直线，参数更少', '不能使用类别先验', '不需要训练数据']"
  :answer="0"
  explanation="类别专属协方差保留二次项，提高表达能力也提高估计方差。"
/>

## 5. 参数怎样估计

令 $n_k$ 为类别 $k$ 样本数：

$$
\hat\pi_k=\frac{n_k}{n},
\qquad
\hat{\boldsymbol\mu}_k
=\frac1{n_k}\sum_{i:y_i=k}\mathbf x_i.
$$

QDA 的 MLE 协方差：

$$
\hat{\boldsymbol\Sigma}_k
=\frac1{n_k}
\sum_{i:y_i=k}
(\mathbf x_i-\hat{\boldsymbol\mu}_k)
(\mathbf x_i-\hat{\boldsymbol\mu}_k)^\top.
$$

LDA 的共享协方差可池化各类类内散度：

$$
\hat{\boldsymbol\Sigma}
=\frac1n\sum_k
\sum_{i:y_i=k}
(\mathbf x_i-\hat{\boldsymbol\mu}_k)
(\mathbf x_i-\hat{\boldsymbol\mu}_k)^\top.
$$

无偏估计会调整分母；分类 MLE 与统计无偏估计的目标仍需区分。

::: ml-stop 停下来推一推 2
为什么共享协方差要围绕各自类别均值计算，而不是围绕全局均值？全局均值会混入哪部分变化？
:::

<MlQuiz
  question="LDA 的共享协方差主要汇总什么？"
  :options="['类别内部围绕各类均值的散度', '仅包含类别均值之间的距离', '标签编号方差', '测试集误差']"
  :answer="0"
  explanation="它估计类条件噪声结构；围绕全局均值会混入类间均值差。"
/>

## 6. 朴素 Bayes：用强假设换低估计成本

朴素 Bayes 假设给定类别后各特征条件独立：

$$
p(\mathbf x\mid y=k)
=\prod_{j=1}^{d}p(x_j\mid y=k).
$$

对数分数变为

$$
\log p(y=k\mid\mathbf x)
=\text{常数}+\log\pi_k
+\sum_j\log p(x_j\mid y=k).
$$

对词袋文本，多项式朴素 Bayes 使用词计数。带平滑的词概率：

$$
\hat p(w_j\mid y=k)
=\frac{N_{kj}+\alpha}
{\sum_\ell N_{k\ell}+\alpha V}.
$$

即使独立假设不真实，决策边界仍可能好用；但相关重复特征会让证据被重复计算，概率往往不校准。

<MlQuiz
  question="朴素 Bayes 的“朴素”具体指什么？"
  :options="['类别只有两个', '给定类别后特征条件独立', '参数都等于零', '不使用 Bayes 公式']"
  :answer="1"
  explanation="条件独立把高维联合密度分解成一维分布乘积，大幅降低估计难度。"
/>

## 7. 先验变化会移动边界

二分类选择 1 的条件是

$$
\log\frac{p(\mathbf x\mid y=1)}
{p(\mathbf x\mid y=0)}
>
\log\frac{\pi_0}{\pi_1}.
$$

若部署环境正类比例与训练不同，仅调整先验项就可能移动决策边界；但这要求类条件分布相对稳定。若 $p(\mathbf x\mid y)$ 也变了，仅改阈值不够。

::: ml-stop 停下来推一推 3
在类条件密度不变时，部署正类先验升高，分类器应更容易还是更难预测正类？从右侧阈值判断。
:::

<MlQuiz
  question="LDA/QDA 能直接生成新特征样本的原因是什么？"
  :options="['它们建模了 p(x|y) 与 p(y)', '它们只建模分隔超平面', '它们不含概率', '它们自动知道因果机制']"
  :answer="0"
  explanation="先采样类别，再从对应类条件高斯采样 x，即可按模型生成；生成质量取决于假设。"
/>

## 8. 动手练习：同一数据的三条边界

生成两个二维高斯类别：

1. 先设相同协方差，对比逻辑回归与 LDA；
2. 再让协方差不同，对比 LDA 与 QDA；
3. 减少样本量，观察 QDA 边界波动；
4. 对 QDA 协方差做收缩；
5. 改变类别先验，画边界移动；
6. 报告准确率之外的概率校准曲线。

<MlLessonProgress
  storage-key="23-generative-classifiers"
  next-href="/courses/machine-learning/24-em"
  next-label="下一章：看不见的变量怎样被估计"
/>
